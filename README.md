# CO-Calib

English | [简体中文](README.zh-CN.md)

<div align="left">
        <!-- <h3>Observation Quality Matters: Robust Multi-Fisheye Calibration via Failure-Oriented Analysis</h3> -->
        <a href="https://uav.hkust.edu.hk/current-members/" target="_blank">Peize Liu</a>,
        Zhe Tong,
        <a href="https://chen-albert-feng.github.io/AlbertFeng.github.io/" target="_blank">Chen Feng</a><sup>†</sup>, and
        <a href="https://uav.hkust.edu.hk/group/" target="_blank">Shaojie Shen</a>
    <p>
        <h45>
            HKUST Aerial Robotics Group &nbsp;&nbsp;
            <br>
        </h5>
        <sup>†</sup>Corresponding Author
    </p>
    <a href='https://arxiv.org/abs/2607.05777'><img src='https://img.shields.io/badge/arXiv-COCalib-red' alt='arxiv'></a>
    <a href='https://peize-liu.github.io/CO-Calib-IO/'><img src='https://img.shields.io/badge/Project_Page-COCalib-green' alt='Project Page'></a>
    <a href="https://www.bilibili.com/video/BV1AqMi6AERS/?spm_id_from=333.1387.list.card_archive.click&vd_source=0af61c122e5e37c944053b57e313025a"><img alt="Bilibili" src="https://img.shields.io/badge/Video-Bilibili-blue"/></a>
    <a href="https://www.youtube.com/watch?v=EskcF2ODnSw"><img alt="Youtube" src="https://img.shields.io/badge/Video-Youtube-red"/></a>
</div>

CO-Calib supports rigs with any number of cameras. It automatically detects the input format, detects the calibration target, runs Datawash, generates a standard ROS1 calibration bag, and launches Kalibr calibration.

The default NN-Detector uses Task13 weights and refines its coordinates on the original image with `cornerSubPix` (window radius 5, 30 iterations, epsilon 0.001). Datawash and calibration use the same refined points and original network confidence. Selection follows the public CO-Calib implementation. Kalibr retains its random processing order and internal restart policy, with no fixed seed or outer failure retries.

Users only need to provide:

- An image sequence, ROS1 bag, or ROS2 bag
- `rig.yaml`
- `datawash.yaml`
- `target.yaml`
- The detector and compute device

## 1. Installation

Requirements: Linux, Conda, and Docker. NN-Detector GPU inference also requires a working NVIDIA driver.

```bash
cd Opensource
conda env create -f environment.yml
conda activate omnicalib-open
```

Pull the provided Kalibr image. CO-Calib uses this image directly:

```bash
docker pull hkustswarm/co-calib:v1.0
```

Regular users do not need to compile Kalibr, retag the image, or mount local backend code.

## 2. One-command Calibration

### NN-Detector with GPU

This is the recommended configuration:

```bash
omnicalib run \
  --input /path/to/sequence_or_bag \
  --rig /path/to/rig.yaml \
  --datawash /path/to/datawash.yaml \
  --target /path/to/target.yaml \
  --detector NN-Detector \
  --device gpu
```

If the CUDA provider is unavailable or fails to initialize, OmniCalib automatically falls back to CPU and continues the same run.

To explicitly use CPU inference:

```bash
omnicalib run \
  --input /path/to/sequence_or_bag \
  --rig /path/to/rig.yaml \
  --datawash /path/to/datawash.yaml \
  --target /path/to/target.yaml \
  --detector NN-Detector \
  --device cpu
```

`--device auto` is the default. It tries GPU first and falls back to CPU. `NN-Detector` is also the default detector, so both options may be omitted.

### ACV-Detector

```bash
omnicalib run \
  --input /path/to/sequence_or_bag \
  --rig /path/to/rig.yaml \
  --datawash /path/to/datawash.yaml \
  --target /path/to/target.yaml \
  --detector ACV-Detector \
  --device cpu
```

The selected detector is used consistently by both Datawash and final calibration. ACV-Detector runs on CPU inside the Kalibr container.

By default, results are written beside the input as `<input-name>_omnicalib/`. Use `--output /path/to/output` to select another directory. Add `--overwrite` to replace an existing result directory.

## 3. Input Data

OmniCalib automatically recognizes the following formats from `--input`.

### Image sequence

```text
sequence/
├── cam0/
│   ├── images/
│   │   ├── 000000.png
│   │   └── ...
│   └── timestamps.csv
├── cam1/
│   ├── images/
│   └── timestamps.csv
└── camN/
    ├── images/
    └── timestamps.csv
```

Each camera requires an image directory and a `timestamps.csv` file:

```csv
frame_id,timestamp_ns,filename
0,1700000000000000000,000000.png
1,1700000000100000000,000001.png
```

PNG, JPEG, BMP, and TIFF images are supported. Cameras do not need identical frame rates. Frames are grouped using the synchronization tolerance in `rig.yaml`.

### ROS1 bag

Pass one `.bag` file to `--input`. Both `sensor_msgs/Image` and `sensor_msgs/CompressedImage` are supported.

### ROS2 bag

Pass a rosbag2 directory containing `metadata.yaml` to `--input`. Both `sensor_msgs/Image` and `sensor_msgs/CompressedImage` are supported.

`examples/stereo_10frames/` contains equivalent image-sequence, ROS1, and ROS2 examples with 10 frames per camera.

## 4. Configuration

### rig.yaml

`rig.yaml` defines camera count, topics, sequence directories, and camera models:

```yaml
sync_tolerance_ms: 10.0
cameras:
  - id: cam0
    topic: /camera_0/image_compressed
    model: omni-none
    directory: cam0
    frame_id: camera_0
  - id: cam1
    topic: /camera_1/image_compressed
    model: omni-none
    directory: cam1
    frame_id: camera_1
```

Add more `cameras` entries for larger rigs. Every `id` and `topic` must be unique, and `directory` must match the corresponding image-sequence directory. Supported models:

- `pinhole-radtan`
- `pinhole-equi`
- `pinhole-fov`
- `omni-none`
- `omni-radtan`
- `eucm-none`
- `ds-none`

### datawash.yaml

```yaml
detector_confidence: 0.99
sample_min_detection_points: 12

selection:
  anchor:
    radial_span: 0.25
    iso: 0.50
    budget: 0
  covisible:
    radial_span: 0.00
    iso: 0.30
    budget: 100
  mono_fill:
    radial_span: 0.20
    iso: 0.30
    budget: 0
```

- `detector_confidence`: Minimum confidence for a detected point.
- `sample_min_detection_points`: Minimum number of valid points required for an image to enter selection.
- `radial_span`: Minimum normalized radial span of the calibration target in the valid image region.
- `iso`: Projection-Jacobian isotropy threshold. Values closer to `1` are stricter.
- `budget`: Per-camera limit for anchor and mono fill, and per-camera-pair limit for covisible. `0` disables the hard limit.

### target.yaml

The current detectors target a 6 x 6 AprilGrid:

```yaml
target_type: aprilgrid
tagCols: 6
tagRows: 6
tagSize: 0.055
tagSpacing: 0.3
```

`tagSize` and `tagSpacing` must match the printed target.

Editable example configurations are provided in:

- `configs/rig_stereo.yaml`
- `configs/datawash.yaml`
- `configs/target_aprilgrid_6x6.yaml`

## 5. Output

```text
<input-name>_omnicalib/
├── datawash/
│   ├── calibration_clean.bag
│   ├── selected_roles.csv
│   └── summary.json
├── kalibr/
│   ├── calibration-camchain.yaml
│   ├── calibration-results-cam.txt
│   └── calibration-report-cam.pdf
└── summary.json
```

`selected_roles.csv` records anchor, covisible, and mono-fill selections. Its `active_cameras` field lists the cameras actually used at each timestamp.

## 6. Extrinsic Visualization

The browser visualizer in `visualization/` displays multi-camera and camera-IMU extrinsics from Kalibr's `camchain.yaml` or `camchain-imucam.yaml`. All YAML parsing happens locally in the browser and no calibration data is uploaded.

From the `Opensource/` repository root, create the visualization environment and start the server:

```bash
cd visualization
conda env create -f environment.yml
conda activate kalibr-visualizer
python -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/` and select or drag the generated file:

```text
<input-name>_omnicalib/kalibr/calibration-camchain.yaml
# or a Kalibr camera-IMU result
calibration-camchain-imucam.yaml
```

The visualizer shows camera frustums, camera axes, adjacent baselines, distances, a world grid, and a pose table. If camera nodes contain `T_cam_imu`, it also displays the IMU body, native IMU axes, reference-camera link, relative rotation, and `timeshift_cam_imu` in milliseconds. It prefers `cam0` as the IMU reference and supports any contiguous `cam0`, `cam1`, ..., `camN` chain whose cameras after `cam0` provide a 4 x 4 `T_cn_cnm1` transform.

Controls:

- Left drag: orbit
- Right drag or Alt + drag: pan
- Mouse wheel: zoom
- `Reset view`: restore the default view
- `Fit rig`: fit all cameras into the viewport

Stop the local server with `Ctrl+C`.

## 7. Citation

Please cite our paper if you use this project in your research:
```
@article{liu2026observation,
  title={Observation Quality Matters: Robust Multi-Fisheye Calibration via Failure-Oriented Analysis},
  author={Liu, Peize and Tong, Zhe and Feng, Chen and Shen, Shaojie},
  journal={arXiv preprint arXiv:2607.05777},
  year={2026}
}
```

## 8. License

Unless a file or directory states otherwise, project-owned CO-Calib source
code, documentation, and the self-developed
`models/nn_detector_aprilgrid_6x6.onnx` weights are distributed under the
BSD-4-Clause license in [`LICENSE`](LICENSE). Model ownership and scope are
documented in [`models/MODEL_CARD.md`](models/MODEL_CARD.md).

This repository distributes a modified copy of Kalibr under `vendor/kalibr`.
Redistributions must preserve the original Kalibr copyright notices,
conditions, disclaimer, advertising acknowledgement, and non-endorsement
clause. In particular, advertising materials that mention features or use of
this software must include:

> This product includes software developed by the Autonomous Systems Lab and Skybotix AG.

The Kalibr `incremental_calibration` and `incremental_calibration_python`
components declare LGPLv3 terms and remain under LGPLv3. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md),
[`vendor/kalibr/LICENSE`](vendor/kalibr/LICENSE), and [`LICENSES/`](LICENSES/)
for the complete distribution notices and component-specific terms.

## 7. Automatic Lens-border Masks

Generate one mask per camera from a moving calibration sequence, ROS1 bag, or ROS2 bag:

```bash
omnicalib mask \
  --input /path/to/sequence_or_bag \
  --rig /path/to/rig.yaml \
  --output /path/to/masks \
  --max-frames 300 \
  --dark-threshold 45 \
  --texture-threshold 5 \
  --persistence 0.95 \
  --feature-rate 0.01 \
  --margin 2
```

The tool samples frames uniformly across each camera stream. A pixel is excluded if it is dark and has low local texture in at least 95% of sampled frames, belongs to a region connected to the image border, and lacks repeated Shi-Tomasi corner detections. A two-pixel margin extends the excluded border while protecting repeated corners. No GPU or new dependencies are required.

Outputs include `camN_mask.png` (uint8: **255 = keep, 0 = exclude**), per-camera overlays, temporal mean images, dark-frequency maps, `overview.jpg`, and `summary.json`. Masks preserve the original image dimensions. Use an empty output directory. At least 10 frames per camera are required; 300 varied frames are recommended.

`--dark-threshold` is an 8-bit grayscale threshold; `--texture-threshold` limits grayscale standard deviation in a 7 x 7 neighborhood. `--feature-rate` protects neighborhoods with corners in at least this fraction of frames. Increasing the brightness/texture thresholds or decreasing persistence produces a less conservative mask. Review the overlays, especially when moving masks to another recording. A stationary dark occluder can look like a lens border; missing corners alone do not prove an area is unusable.

This command exports masks; calibration and Datawash do not automatically consume these files. Feature trackers should use the PNG as their detection mask. Zeroing pixels is useful for previews but can introduce artificial edges at the mask boundary.
