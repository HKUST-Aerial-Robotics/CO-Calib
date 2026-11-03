#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/opt/omnicalib/src")

from omnicalib_open.cache import normalize_detector_name
from omnicalib_open.kalibr_adapter import export_kalibr_npz
from omnicalib_open.rig import load_rig, require_kalibr_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-camera Kalibr with NN-Detector or ACV-Detector")
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--rig", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--detector", type=normalize_detector_name, default="nn")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    rig = load_rig(args.rig)
    models = require_kalibr_models(rig)
    if args.detector == "nn" and args.cache is None:
        raise ValueError("--cache is required when --detector nn")
    if args.detector == "acv" and args.cache is not None:
        raise ValueError("--cache is only valid when --detector nn")
    args.output.mkdir(parents=True, exist_ok=True)
    kalibr_executable = shutil.which("kalibr_calibrate_cameras")
    if kalibr_executable is None:
        candidates = (
            Path("/catkin_ws/devel/.private/kalibr/lib/kalibr/kalibr_calibrate_cameras"),
            Path("/catkin_ws/src/kalibr/aslam_offline_calibration/kalibr/python/kalibr_calibrate_cameras"),
        )
        kalibr_executable = next((str(path) for path in candidates if path.is_file()), None)
    if kalibr_executable is None:
        raise FileNotFoundError("kalibr_calibrate_cameras is not installed in the image")

    help_result = subprocess.run(
        [kalibr_executable, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = f"{help_result.stdout}\n{help_result.stderr}"
    detector_arguments: list[str] = []
    if args.detector == "nn":
        if "--precomputed-detections" not in help_text:
            raise RuntimeError("This Kalibr image does not support NN-Detector caches")
        converted_cache = export_kalibr_npz(args.cache, rig, Path("/tmp/omnicalib_observations.npz"))
        manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
        confidence = float(manifest["confidence_threshold"])
        confidence_flag = "--nn-conf-thr" if "--nn-conf-thr" in help_text else "--fisheye-conf-thr"
        detector_arguments = ["--precomputed-detections", str(converted_cache)]
        if confidence_flag in help_text:
            detector_arguments.extend([confidence_flag, str(confidence)])
        minimum_flag = "--nn-min-corners" if "--nn-min-corners" in help_text else "--fisheye-min-corners"
        if minimum_flag in help_text:
            detector_arguments.extend([minimum_flag, "4"])

    output_arguments = ["--dont-show-report"] if "--dont-show-report" in help_text else []

    command = [
        kalibr_executable,
        "--bag",
        str(args.bag.resolve()),
        "--topics",
        *[camera.topic for camera in rig.cameras],
        "--models",
        *models,
        "--target",
        str(args.target.resolve()),
        *detector_arguments,
        *output_arguments,
        *args.extra,
    ]
    subprocess.run(command, cwd=args.output, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
