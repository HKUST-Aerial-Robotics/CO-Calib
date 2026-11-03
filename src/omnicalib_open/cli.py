from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import normalize_detector_name
from .nn_detector import normalize_device
from .orchestrator import DEFAULT_KALIBR_IMAGE, run_full_pipeline


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path, help="sequence directory, ROS1 bag, or ROS2 bag")
    parser.add_argument("--rig", required=True, type=Path, help="rig YAML containing camera models")
    parser.add_argument("--datawash", required=True, type=Path, help="Datawash YAML")
    parser.add_argument("--target", required=True, type=Path, help="Kalibr target YAML")
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory; defaults to <input-name>_omnicalib beside the input",
    )
    parser.add_argument(
        "--detector",
        type=normalize_detector_name,
        default="nn",
        metavar="{NN-Detector,ACV-Detector}",
        help="detector used by both Datawash and calibration (default: NN-Detector)",
    )
    parser.add_argument(
        "--device",
        type=normalize_device,
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="NN-Detector device; GPU failure automatically falls back to CPU (default: auto)",
    )
    parser.add_argument("--nn-model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--kalibr-image", default=DEFAULT_KALIBR_IMAGE, help=argparse.SUPPRESS)
    parser.add_argument("--kalibr-detector-processes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")


def _run_from_args(args: argparse.Namespace) -> int:
    summary = run_full_pipeline(
        source=args.input,
        rig_path=args.rig,
        datawash_path=args.datawash,
        target_path=args.target,
        output_dir=args.output,
        detector=args.detector,
        nn_model=args.nn_model,
        kalibr_image=args.kalibr_image,
        kalibr_detector_processes=args.kalibr_detector_processes,
        device=args.device,
        overwrite=args.overwrite,
    )
    providers = summary["datawash"].get("execution_providers", [])
    actual_device = "gpu" if "CUDAExecutionProvider" in providers else "cpu"
    public_result = {
        "status": "completed",
        "detector": "NN-Detector" if summary["detector"] == "nn" else "ACV-Detector",
        "input_type": summary["input_type"],
        "device": actual_device if summary["detector"] == "nn" else "cpu",
        "clean_ros1_bag": summary["datawash"]["clean_ros1_bag"],
        "calibration_output": summary["kalibr_output"],
        "output": summary["output"],
    }
    print(json.dumps(public_result, indent=2, sort_keys=True))
    return 0


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run detector, Datawash, and Kalibr")
    _add_run_arguments(parser)
    return _run_from_args(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    from .masking import add_arguments, generate

    parser = argparse.ArgumentParser(prog="omnicalib")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_arguments(subparsers.add_parser("run", help="run detector, Datawash, and Kalibr"))
    add_arguments(subparsers.add_parser("mask", help="generate persistent dark-border masks"))
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_from_args(args)
    if args.command == "mask":
        return generate(args)
    raise RuntimeError(f"Unsupported command: {args.command}")
