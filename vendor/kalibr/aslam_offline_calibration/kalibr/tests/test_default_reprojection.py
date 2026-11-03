#!/usr/bin/env python3
"""Run inside the Kalibr image to check default and rebuilt residual policies."""
import importlib
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest

import aslam_cv as acv
import aslam_cv_backend as acvb
import numpy as np
import sm
from kalibr_common import ConfigReader


calibration = importlib.import_module('kalibr_camera_calibration.CameraCalibrator')


class DefaultReprojectionTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        target_path = Path(self.folder.name) / 'target.yaml'
        target_path.write_text('target_type: aprilgrid\ntagRows: 6\ntagCols: 6\ntagSize: 0.055\ntagSpacing: 0.3\n')
        target = ConfigReader.CalibrationTargetParameters(str(target_path))
        self.camera = calibration.CameraGeometry(acvb.DistortedOmni, target, SimpleNamespace(topic='/cam0'))
        self.camera.geometry.projection().setParameters(np.array([0.8, 300., 300., 640., 360.]))
        self.camera.geometry.projection().distortion().setParameters(np.zeros(4))
        self.camera.isGeometryInitialized = True
        self.observation = acv.GridCalibrationTargetObservation(self.camera.ctarget.detector.target())
        self.observation.setImage(np.zeros((720, 1280), dtype=np.uint8))
        self.observation.updateImagePoint(0, np.array([620., 350.]))
        self.observation.updateImagePoint(1, np.array([650., 350.]))
        transform = np.eye(4)
        transform[2, 3] = -1.
        self.pose = sm.Transformation(transform)

    def batch(self, **kwargs):
        calibrator = calibration.CameraCalibration([self.camera], [], **kwargs)
        batch = calibration.CalibrationTargetOptimizationProblem.fromTargetViewObservations(
            calibrator.cameras, calibrator.target, calibrator.baselines, 0.0,
            self.pose, [(0, self.observation)],
            useBlakeZissermanMest=calibrator.useBlakeZissermanMest,
            reprojectionMEstimator=calibrator.reprojectionMEstimator,
            huberWidth=calibrator.huberWidth)
        return calibrator, batch

    def test_default_and_corner_removal(self):
        calibrator, batch = self.batch()
        self.assertFalse(calibrator.useBlakeZissermanMest)
        self.assertEqual(calibrator.reprojectionMEstimator, 'huber')
        self.assertEqual(calibrator.huberWidth, 1.0)
        for error in batch.rerrs[0][:2]:
            self.assertIn('Huber', error.getMEstimatorName())
            self.assertAlmostEqual(error.getMEstimatorWeight(4.0), 0.5)
        rebuilt = calibration.removeCornersFromBatch(batch, [(0, [0])])
        self.assertIsNone(rebuilt.rerrs[0][0])
        self.assertAlmostEqual(rebuilt.rerrs[0][1].getMEstimatorWeight(4.0), 0.5)

    def test_rebuild_preserves_explicit_backend_policy(self):
        _, batch = self.batch(huberWidth=2.0)
        rebuilt = calibration.removeCornersFromBatch(batch, [(0, [0])])
        self.assertEqual(rebuilt.huberWidth, 2.0)
        self.assertAlmostEqual(rebuilt.rerrs[0][1].getMEstimatorWeight(16.0), 0.5)

    def test_parser_has_no_policy_fields(self):
        backend = runpy.run_path('/catkin_ws/src/kalibr/aslam_offline_calibration/kalibr/python/kalibr_calibrate_cameras',
                                 run_name='_kalibr_default_test')
        previous = sys.argv
        try:
            sys.argv = ['kalibr_calibrate_cameras', '--models', 'omni-radtan',
                        '--topics', '/cam0', '--target', str(Path(self.folder.name) / 'target.yaml')]
            args = backend['parseArgs']()
            self.assertFalse(hasattr(args, 'doHuber'))
            self.assertFalse(hasattr(args, 'huberWidth'))
            self.assertFalse(args.noShuffle)
        finally:
            sys.argv = previous


if __name__ == '__main__':
    unittest.main()
