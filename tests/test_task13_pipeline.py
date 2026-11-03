"""Checks for original-image refinement and shared calibration observations."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from omnicalib_open.cache import read_cache, write_cache
from omnicalib_open.kalibr_adapter import export_kalibr_npz
from omnicalib_open.models import CameraSpec, DatawashConfig, DetectorOutput, FrameGroup, FrameRecord, RigSpec
from omnicalib_open.orchestrator import DEFAULT_KALIBR_IMAGE, _run_kalibr, bundled_nn_model
from omnicalib_open.pipeline import detect_groups, run_datawash
from omnicalib_open.preprocess import compute_resize_plan
from omnicalib_open.subpixel import SUBPIXEL, refine_points


class Task13PipelineTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((80, 120, 3), np.uint8)
        self.image[24:, 24:] = 255
        self.rig = RigSpec((CameraSpec("cam0", "/cam0", "cam0", "cam0", "omni-radtan"),), 10_000_000)

    def test_subpixel_improves_corner_and_preserves_invalid_points(self):
        points = np.array([[22.8, 25.2], [1, 1], [np.nan, 10]], np.float32)
        original = points.copy()
        refined = refine_points(self.image, points)
        self.assertLess(np.linalg.norm(refined[0] - [23.5, 23.5]), 0.2)
        np.testing.assert_array_equal(refined[1:], points[1:])
        np.testing.assert_array_equal(points, original)

    def test_refinement_before_cache_and_same_coordinates_exported(self):
        ok, encoded = cv2.imencode(".png", self.image)
        self.assertTrue(ok)
        frame = FrameRecord("cam0", 0, 123, encoded.tobytes(), "png")
        groups = [FrameGroup(0, 123, {"cam0": frame})]
        xy = np.tile([22.8, 25.2], (144, 1)).astype(np.float32)
        conf = np.linspace(0.5, 1, 144, dtype=np.float32)
        plan = compute_resize_plan(80, 120, 720, 1280)

        class Detector:
            def detect(self, image):
                return DetectorOutput(xy / [plan.scale_x, plan.scale_y], conf)

        records = detect_groups(groups, detector=Detector(), config=DatawashConfig())
        np.testing.assert_allclose(records[0].points_xy, refine_points(self.image, xy), atol=1e-5)
        np.testing.assert_array_equal(records[0].confidence, conf)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = write_cache(root / "cache", records, model_path=bundled_nn_model(),
                                confidence_threshold=.99, camera_ids=["cam0"], subpixel=SUBPIXEL)
            manifest, loaded = read_cache(cache, expected_model=bundled_nn_model())
            self.assertEqual(manifest["subpixel"], SUBPIXEL)
            dest = export_kalibr_npz(cache, self.rig, root / "kalibr.npz")
            with np.load(dest) as arrays:
                np.testing.assert_array_equal(arrays["cam0__xy_px"][0],
                                              loaded[0].points_xy.reshape(36, 4, 2)[:, ::-1])
                np.testing.assert_array_equal(arrays["cam0__conf"][0], conf.reshape(36, 4)[:, ::-1])
            # A Task13 cache without refinement metadata must not silently bypass refinement.
            raw_cache = write_cache(root / "raw", records, model_path=bundled_nn_model(),
                                    confidence_threshold=.99, camera_ids=["cam0"])
            with patch("omnicalib_open.pipeline.load_frame_groups", return_value=groups):
                with self.assertRaisesRegex(ValueError, "subpixel refinement"):
                    run_datawash(source=root, rig=self.rig, config=DatawashConfig(),
                                 model_path=bundled_nn_model(), output_dir=root / "out", input_cache=raw_cache)

    def test_calibration_keeps_single_invocation_without_seed_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("omnicalib_open.orchestrator.subprocess.run") as run:
                _run_kalibr(detector="nn", clean_bag=root / "clean.bag", selected_cache=root / "cache",
                            rig_path=root / "rig.yaml", target_path=root / "target.yaml",
                            output_dir=root / "out", docker_image=DEFAULT_KALIBR_IMAGE)
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(DEFAULT_KALIBR_IMAGE, "hkustswarm/co-calib:v1.0")
            self.assertEqual(command[command.index(DEFAULT_KALIBR_IMAGE) + 1:], [
                "omnicalib-calibrate", "--detector", "nn", "--bag", "/output/calibration.bag",
                "--rig", "/input/rig.yaml", "--target", "/input/target.yaml",
                "--output", "/output", "--cache", "/input/detections.detcache",
            ])
            mounts = [command[i + 1] for i, value in enumerate(command) if value == "-v"]
            self.assertEqual(len(mounts), 5)
            self.assertFalse(any("/opt/omnicalib" in mount for mount in mounts))
            self.assertNotIn("--no-shuffle", command)
            self.assertFalse(any("seed" in item.lower() for item in command))

    def test_container_wrapper_passes_only_input_and_output_options(self):
        source = Path(__file__).resolve().parents[1] / "docker/kalibr/run_calibration.py"
        spec = importlib.util.spec_from_file_location("cocalib_container_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            argv = ["omnicalib-calibrate", "--bag", str(Path(folder) / "input.bag"),
                    "--rig", "rig.yaml", "--target", "target.yaml", "--detector", "acv",
                    "--output", folder]
            with patch.object(module, "load_rig", return_value=self.rig), \
                 patch.object(module.shutil, "which", return_value="/bin/kalibr"), \
                 patch.object(module.sys, "argv", argv), \
                 patch.object(module.subprocess, "run") as run:
                run.return_value = SimpleNamespace(stdout="--dont-show-report", stderr="")
                self.assertEqual(module.main(), 0)
                command = run.call_args.args[0]
                self.assertEqual(command, [
                    "/bin/kalibr", "--bag", str(Path(folder) / "input.bag"),
                    "--topics", "/cam0", "--models", "omni-radtan", "--target",
                    str(Path("target.yaml").resolve()), "--dont-show-report",
                ])
                self.assertEqual(run.call_count, 2)  # Help probe plus one calibration invocation.
                self.assertNotIn("--no-shuffle", command)


if __name__ == "__main__":
    unittest.main()
