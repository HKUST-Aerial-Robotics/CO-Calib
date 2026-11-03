# NN AprilGrid Detector

- Default file: `nn_detector_task13_aprilgrid_6x6.onnx` (FD Task13)
- SHA256: `df230cc4649c718e48870391ca0a0e91cca5d9802ca8746c9bef532824107daf`
- Postprocessing: coordinates are mapped to the original image, then refined with OpenCV `cornerSubPix` (window radius 5, 30 iterations, epsilon 0.001). Network confidence is retained. Frame selection and calibration use the same refined coordinates.
- The previous `nn_detector_aprilgrid_6x6.onnx` remains available for historical reproduction; it is not the default.
- Purpose: detect ordered corners of a 6 x 6 AprilGrid
- Copyright: Copyright (c) 2026, HKUST Aerial Robotics Group and CO-Calib contributors
- License: BSD-4-Clause, as provided in the repository's top-level `LICENSE`
- Origin: the architecture, training procedure, and distributed model weights are original work developed by the CO-Calib authors

The model is distributed with CO-Calib and may be redistributed and modified
under the same BSD-4-Clause terms as the project-owned source code.
