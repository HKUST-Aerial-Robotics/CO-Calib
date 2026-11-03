from __future__ import print_function #handle print in 2.x python
import sm
from sm import PlotCollection
from kalibr_common import ConfigReader as cr
import aslam_cv as acv
import aslam_cameras_april as acv_april
import aslam_cv_backend as acvb
import aslam_backend as aopt
import incremental_calibration as ic
import kalibr_camera_calibration as kcc

from matplotlib.backends.backend_pdf import PdfPages
import mpl_toolkits.mplot3d.axes3d as p3
import cv2
import numpy as np
import pylab as pl
import math
import gc
import sys
import os

np.set_printoptions(suppress=True, precision=8)

#DV group IDs
CALIBRATION_GROUP_ID = 0
TRANSFORMATION_GROUP_ID = 1
LANDMARK_GROUP_ID = 2


def _use_default_pose_on_pnp_fail():
    return os.environ.get("KALIBR_USE_DEFAULT_POSE_ON_PNP_FAIL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _observation_corner_count(obs):
    return int(obs.getCornersImageFrame().shape[0])


def _seed_manual_omni_intrinsics(camera_geometry, observations):
    if not observations:
        return False
    proj = camera_geometry.geometry.projection().getParameters().flatten()
    if proj.shape[0] != 5:
        return False
    width = float(observations[0].imCols())
    height = float(observations[0].imRows())
    focal_seed = 0.55 * width
    proj_seed = np.array(
        [1.0, focal_seed, focal_seed, 0.5 * (width - 1.0), 0.5 * (height - 1.0)],
        dtype=np.float64,
    )
    dist_seed = np.zeros_like(
        camera_geometry.geometry.projection().distortion().getParameters().flatten(),
        dtype=np.float64,
    )
    camera_geometry.geometry.projection().setParameters(proj_seed)
    camera_geometry.geometry.projection().distortion().setParameters(dist_seed)
    sm.logWarn(
        "Falling back to manual omni intrinsic seed for topic {0}: {1}".format(
            camera_geometry.dataset.topic, proj_seed.tolist()
        )
    )
    return True


def _make_reprojection_mest(kind, huber_width=1.0):
    kind = (kind or 'none').strip().lower()
    if kind in ('none', '', 'l2'):
        return None
    if kind in ('blake', 'blake-zisserman', 'blakezisserman'):
        return aopt.BlakeZissermanMEstimator(2.0)
    if kind == 'huber':
        return aopt.HuberMEstimator(float(huber_width))
    raise RuntimeError('Unsupported reprojection M-estimator: {0}'.format(kind))


def _camera_state_is_finite(camera_geometry):
    try:
        proj = camera_geometry.geometry.projection().getParameters().flatten()
        dist = camera_geometry.geometry.projection().distortion().getParameters().flatten()
    except Exception:
        return False
    return np.all(np.isfinite(proj)) and np.all(np.isfinite(dist))


def _snapshot_camera_state(camera_geometry):
    return (
        np.array(camera_geometry.geometry.projection().getParameters().flatten(), dtype=np.float64),
        np.array(camera_geometry.geometry.projection().distortion().getParameters().flatten(), dtype=np.float64),
    )


def _restore_camera_state(camera_geometry, snapshot):
    proj, dist = snapshot
    camera_geometry.geometry.projection().setParameters(proj)
    camera_geometry.geometry.projection().distortion().setParameters(dist)


def _transformation_is_finite(T):
    try:
        return np.all(np.isfinite(np.asarray(T.T(), dtype=np.float64)))
    except Exception:
        return False

class OptimizationDiverged(Exception):
    pass

class CameraGeometry(object):
    def __init__(self, cameraModel, targetConfig, dataset, geometry=None, verbose=False):
        self.dataset = dataset
        
        self.model = cameraModel
        if geometry is None:
            self.geometry = cameraModel.geometry()
        
        if not type(self.geometry) == cameraModel.geometry:
            raise RuntimeError("The type of geometry passed in \"%s\" does not match the model type \"%s\"" % (type(geometry),type(cameraModel.geometry)))
        
        #create the design variables
        self.dv = cameraModel.designVariable(self.geometry)
        self.setDvActiveStatus(True, True, False)
        self.isGeometryInitialized = False

        #create target detector
        self.ctarget = TargetDetector(targetConfig, self.geometry, showCorners=verbose)

    def setDvActiveStatus(self, projectionActive, distortionActive, shutterActice):
        self.dv.projectionDesignVariable().setActive(projectionActive)
        self.dv.distortionDesignVariable().setActive(distortionActive)
        self.dv.shutterDesignVariable().setActive(shutterActice)

    def initGeometryFromObservations(self, observations):
        init_observations = [
            obs for obs in observations
            if _observation_corner_count(obs) >= 6
        ]
        if len(init_observations) != len(observations):
            sm.logWarn(
                "Skipping {0} observations with fewer than 6 corners during intrinsic initialization for topic {1}".format(
                    len(observations) - len(init_observations), self.dataset.topic))
        if len(init_observations) == 0:
            sm.logError("No valid observations (>=6 corners) available for topic {0}".format(self.dataset.topic))
            return False

        #obtain focal length guess
        try:
            success = self.geometry.initializeIntrinsics(init_observations)
        except RuntimeError as exc:
            if self.model == acvb.DistortedOmni and _seed_manual_omni_intrinsics(self, init_observations):
                sm.logWarn(
                    "initializeIntrinsics threw for topic {0}; continuing from manual omni seed. Error: {1}".format(
                        self.dataset.topic, exc
                    )
                )
                success = True
            else:
                raise
        if not success:
            sm.logError("initialization of focal length for cam with topic {0} failed  ".format(self.dataset.topic))
        if not _camera_state_is_finite(self):
            sm.logError("initialization produced non-finite camera parameters for topic {0}".format(self.dataset.topic))
            self.isGeometryInitialized = False
            return False
        last_finite_snapshot = _snapshot_camera_state(self)
        
        #in case of an omni model, first optimize over intrinsics only
        #(--> catch most of the distortion with the projection model)
        if self.model == acvb.DistortedOmni:
            success = kcc.calibrateIntrinsics(self, init_observations, distortionActive=False)
            if not success or not _camera_state_is_finite(self):
                sm.logWarn("initialization of intrinsics for cam with topic {0} produced invalid parameters; restoring last finite state and continuing".format(self.dataset.topic))
                _restore_camera_state(self, last_finite_snapshot)
            else:
                last_finite_snapshot = _snapshot_camera_state(self)
        
        #optimize for intrinsics & distortion    
        success = kcc.calibrateIntrinsics(self, init_observations)
        if not success or not _camera_state_is_finite(self):
            sm.logWarn("initialization of intrinsics for cam with topic {0} produced invalid parameters; restoring last finite state and continuing".format(self.dataset.topic))
            _restore_camera_state(self, last_finite_snapshot)
        else:
            last_finite_snapshot = _snapshot_camera_state(self)
        
        self.isGeometryInitialized = _camera_state_is_finite(self)
        return self.isGeometryInitialized

class TargetDetector(object):
    def __init__(self, targetConfig, cameraGeometry, showCorners=False, showReproj=False, showOneStep=False):
        self.targetConfig = targetConfig
        
        #initialize the calibration target
        targetParams = targetConfig.getTargetParams()
        targetType = targetConfig.getTargetType()

        if targetType == 'checkerboard':
            options = acv.CheckerboardOptions()
            options.filterQuads = True
            options.normalizeImage = True
            options.useAdaptiveThreshold = True        
            options.performFastCheck = False
            options.windowWidth = 5            
            options.showExtractionVideo = showCorners
            
            self.grid = acv.GridCalibrationTargetCheckerboard(targetParams['targetRows'], 
                                                              targetParams['targetCols'], 
                                                              targetParams['rowSpacingMeters'], 
                                                              targetParams['colSpacingMeters'], 
                                                              options)
        elif targetType == 'circlegrid':
            options = acv.CirclegridOptions()
            options.showExtractionVideo = showCorners
            options.useAsymmetricCirclegrid = targetParams['asymmetricGrid']
            
            self.grid = acv.GridCalibrationTargetCirclegrid(targetParams['targetRows'],
                                                           targetParams['targetCols'], 
                                                           targetParams['spacingMeters'], 
                                                           options)
         
        elif targetType == 'aprilgrid':
            options = acv_april.AprilgridOptions()
            #enforce more than one row --> pnp solution can be bad if all points are almost on a line...
            options.minTagsForValidObs = int( np.max( [targetParams['tagRows'], targetParams['tagCols']] ) + 1 )
            options.showExtractionVideo = showCorners
            
            self.grid = acv_april.GridCalibrationTargetAprilgrid(targetParams['tagRows'], 
                                                                 targetParams['tagCols'], 
                                                                 targetParams['tagSize'], 
                                                                 targetParams['tagSpacing'], 
                                                                 options)
        else:
            RuntimeError('Unknown calibration target type!')

        options = acv.GridDetectorOptions() 
        options.imageStepping = showOneStep
        options.plotCornerReprojection = showReproj
        options.filterCornerOutliers = False
        
        self.detector = acv.GridDetector(cameraGeometry, self.grid, options)

class CalibrationTarget(object):
    def __init__(self, target, estimateLandmarks=False):
        self.target = target
        # Create design variables and expressions for all target points.
        P_t_dv = []
        P_t_ex = []
        for i in range(0,self.target.size()):
            p_t_dv = aopt.HomogeneousPointDv(sm.toHomogeneous(self.target.point(i)));
            p_t_dv.setActive(estimateLandmarks)
            p_t_ex = p_t_dv.toExpression()
            P_t_dv.append(p_t_dv)
            P_t_ex.append(p_t_ex)
        self.P_t_dv = P_t_dv
        self.P_t_ex = P_t_ex
    def getPoint(self,i):
        return P_t_ex[i]

class CalibrationTargetOptimizationProblem(ic.CalibrationOptimizationProblem):        
    @classmethod
    def fromTargetViewObservations(cls, cameras, target, baselines, timestamp, T_tc_guess, rig_observations, useBlakeZissermanMest=False, fixIntrinsics=False, reprojectionMEstimator='huber', huberWidth=1.0):
        rval = CalibrationTargetOptimizationProblem()        

        #store the arguements in case we want to rebuild a modified problem
        rval.cameras = cameras
        rval.target = target
        rval.baselines = baselines
        rval.timestamp = timestamp
        rval.T_tc_guess = T_tc_guess
        rval.rig_observations = rig_observations
        rval.fixIntrinsics = fixIntrinsics
        rval.reprojectionMEstimator = 'blake' if useBlakeZissermanMest else reprojectionMEstimator
        rval.huberWidth = huberWidth
        
        # 1. Create a design variable for this pose
        T_target_camera = T_tc_guess
        
        rval.dv_T_target_camera = aopt.TransformationDv(T_target_camera)
        for i in range(0, rval.dv_T_target_camera.numDesignVariables()):
            rval.addDesignVariable( rval.dv_T_target_camera.getDesignVariable(i), TRANSFORMATION_GROUP_ID)
        
        #2. add all baselines DVs
        for baseline_dv in baselines:
            for i in range(0, baseline_dv.numDesignVariables()):
                rval.addDesignVariable(baseline_dv.getDesignVariable(i), CALIBRATION_GROUP_ID)
        
        #3. add landmark DVs
        for p in target.P_t_dv:
            rval.addDesignVariable(p,LANDMARK_GROUP_ID)
        
        #4. add camera DVs
        for camera in cameras:
            if not camera.isGeometryInitialized:
                raise RuntimeError('The camera geometry is not initialized. Please initialize with initGeometry() or initGeometryFromDataset()')
            camera.setDvActiveStatus(not fixIntrinsics, not fixIntrinsics, False)
            rval.addDesignVariable(camera.dv.distortionDesignVariable(), CALIBRATION_GROUP_ID)
            rval.addDesignVariable(camera.dv.projectionDesignVariable(), CALIBRATION_GROUP_ID)
            rval.addDesignVariable(camera.dv.shutterDesignVariable(), CALIBRATION_GROUP_ID)
        
        #4.add all observations for this view
        cams_in_view = set()
        rval.rerrs=dict()
        rerr_cnt=0
        for cam_id, obs in rig_observations:
            camera = cameras[cam_id]
            cams_in_view.add(cam_id)
            
            #add reprojection errors
            #build baseline chain (target->cam0->baselines->camN)                
            T_cam0_target = rval.dv_T_target_camera.expression.inverse()
            T_camN_calib = T_cam0_target
            for idx in range(0, cam_id):
                T_camN_calib =  baselines[idx].toExpression() * T_camN_calib
            
            # \todo pass in the detector uncertainty somehow.
            cornerUncertainty = 1.0
            R = np.eye(2) * cornerUncertainty * cornerUncertainty
            invR = np.linalg.inv(R)
            
            rval.rerrs[cam_id] = list()
            for i in range(0,len(target.P_t_ex)):
                p_target = target.P_t_ex[i]
                valid, y = obs.imagePoint(i)
                if valid:
                    rerr_cnt+=1
                    # Create an error term.
                    rerr = camera.model.reprojectionError(y, invR, T_camN_calib * p_target, camera.dv)
                    rerr.idx = i
                    
                    # add optional reprojection M-estimator
                    mest_kind = 'blake' if useBlakeZissermanMest else reprojectionMEstimator
                    mest = _make_reprojection_mest(mest_kind, huberWidth)
                    if mest is not None:
                        rerr.setMEstimatorPolicy(mest)
                    rval.addErrorTerm(rerr)
                    rval.rerrs[cam_id].append(rerr)
                else:
                    rval.rerrs[cam_id].append(None)

        sm.logDebug("Adding a view with {0} cameras and {1} error terms".format(len(cams_in_view), rerr_cnt))
        return rval

def removeCornersFromBatch(batch, camId_cornerIdList_tuples, useBlakeZissermanMest=False, reprojectionMEstimator='huber', huberWidth=1.0):
    #translate (camid,obs) tuple to dict
    obsdict=dict()
    for cidx, obs in batch.rig_observations:
        obsdict[cidx]=obs
       
    #disable the corners
    hasCornerRemoved=False
    for cidx, removelist in camId_cornerIdList_tuples:
        for corner_id in removelist: 
            obsdict[cidx].removeImagePoint(corner_id)
            hasCornerRemoved=True
    assert hasCornerRemoved, "need to remove at least one corner..."
    
    #rebuild problem
    new_problem = CalibrationTargetOptimizationProblem.fromTargetViewObservations(batch.cameras, 
                                                                                  batch.target, 
                                                                                  batch.baselines, 
                                                                                  batch.timestamp, 
                                                                                  batch.T_tc_guess, 
                                                                                  batch.rig_observations,
                                                                                  useBlakeZissermanMest=useBlakeZissermanMest,
                                                                                  fixIntrinsics=getattr(batch, "fixIntrinsics", False),
                                                                                  reprojectionMEstimator=getattr(batch, "reprojectionMEstimator", reprojectionMEstimator),
                                                                                  huberWidth=getattr(batch, "huberWidth", huberWidth))

    return new_problem
        
class CameraCalibration(object):
    def __init__(self, cameras, baseline_guesses, estimateLandmarks=False, verbose=False, useBlakeZissermanMest=False, fixIntrinsics=False, reprojectionMEstimator='huber', huberWidth=1.0):
        self.cameras = cameras
        self.useBlakeZissermanMest = useBlakeZissermanMest
        self.reprojectionMEstimator = 'blake' if useBlakeZissermanMest else reprojectionMEstimator
        self.huberWidth = huberWidth
        self.fixIntrinsics = fixIntrinsics
        #create the incremental estimator
        self.estimator = ic.IncrementalEstimator(CALIBRATION_GROUP_ID)
        self.linearSolverOptions = self.estimator.getLinearSolverOptions()
        self.optimizerOptions = self.estimator.getOptimizerOptions()
        self.target = CalibrationTarget(cameras[0].ctarget.detector.target(), estimateLandmarks=estimateLandmarks)
        self.initializeBaselineDVs(baseline_guesses)
        #storage for the used views
        self.views = list()
        
    def initializeBaselineDVs(self, baseline_guesses):
        self.baselines = list()
        for baseline_idx in range(0, len(self.cameras)-1): 
            self.baselines.append( aopt.TransformationDv(baseline_guesses[baseline_idx]) )
            
    def getBaseline(self, i):
        return self.baselines[i]
    
    def addTargetView(self, timestamp, rig_observations, T_tc_guess, force=False, allowDiverged=False):
        if T_tc_guess is None:
            if _use_default_pose_on_pnp_fail():
                sm.logWarn("addTargetView: using default identity pose fallback at timestamp {0}".format(timestamp))
                T_tc_guess = sm.Transformation()
            else:
                sm.logWarn("Skipping target view at timestamp {0}: missing initial target pose guess (PnP failed or was rejected)".format(timestamp))
                return False
        if not _transformation_is_finite(T_tc_guess):
            sm.logWarn("Skipping target view at timestamp {0}: non-finite initial target pose guess".format(timestamp))
            return False
        for cam in self.cameras:
            if not _camera_state_is_finite(cam):
                sm.logWarn("Skipping target view at timestamp {0}: camera geometry contains NaN/Inf before batch insertion".format(timestamp))
                return False
        #create the problem for this batch and try to add it 
        batch_problem = CalibrationTargetOptimizationProblem.fromTargetViewObservations(self.cameras, self.target, self.baselines, timestamp, T_tc_guess, rig_observations, useBlakeZissermanMest=self.useBlakeZissermanMest, fixIntrinsics=self.fixIntrinsics, reprojectionMEstimator=self.reprojectionMEstimator, huberWidth=self.huberWidth)
        self.estimator_return_value = self.estimator.addBatch(batch_problem, force)
        
        if self.estimator_return_value.numIterations >= self.optimizerOptions.maxIterations and not allowDiverged:
            sm.logError("Did not converge in maxIterations... restarting...")
            raise OptimizationDiverged
        
        success = self.estimator_return_value.batchAccepted
        if success:
            sm.logDebug("The estimator accepted this batch")
            self.views.append(batch_problem)
        else:
            sm.logDebug("The estimator did not accept this batch")
        return success
