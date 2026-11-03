from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .cache import normalize_detector_name
from .config import load_datawash_config
from .kalibr_detector import extract_acv_detection_cache
from .pipeline import run_datawash
from .rig import load_rig, require_kalibr_models
from .sources import detect_source_type, load_frame_groups


DEFAULT_KALIBR_IMAGE = "hkustswarm/co-calib:v1.0"


def bundled_nn_model() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "nn_detector_task13_aprilgrid_6x6.onnx"


def default_output_dir(source: str | Path) -> Path:
    input_path = Path(source).resolve()
    detect_source_type(input_path)
    name = input_path.stem if input_path.is_file() else input_path.name
    return input_path.parent / f"{name}_omnicalib"


def _mount(source: Path, target: str, *, read_only: bool = False) -> str:
    return f"{source.resolve()}:{target}{':ro' if read_only else ''}"


def _run_kalibr(
    *,
    detector: str,
    clean_bag: Path,
    selected_cache: Path,
    rig_path: Path,
    target_path: Path,
    output_dir: Path,
    docker_image: str,
) -> None:
    detector_name = normalize_detector_name(detector)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        "ROS_HOME=/tmp/.ros",
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "-e",
        "MPLBACKEND=Agg",
        "-v",
        _mount(output_dir, "/output"),
        "-v",
        _mount(clean_bag, "/output/calibration.bag", read_only=True),
        "-v",
        _mount(rig_path, "/input/rig.yaml", read_only=True),
        "-v",
        _mount(target_path, "/input/target.yaml", read_only=True),
    ]
    if detector_name == "nn":
        command.extend(["-v", _mount(selected_cache, "/input/detections.detcache", read_only=True)])
    command.extend(
        [
            str(docker_image),
            "omnicalib-calibrate",
            "--detector",
            detector_name,
            "--bag",
            "/output/calibration.bag",
            "--rig",
            "/input/rig.yaml",
            "--target",
            "/input/target.yaml",
            "--output",
            "/output",
        ]
    )
    if detector_name == "nn":
        command.extend(["--cache", "/input/detections.detcache"])
    mounted_bag_placeholder = output_dir / "calibration.bag"
    try:
        subprocess.run(command, check=True)
    finally:
        if mounted_bag_placeholder.is_file() and mounted_bag_placeholder.stat().st_size == 0:
            mounted_bag_placeholder.unlink()


def run_full_pipeline(
    *,
    source: str | Path,
    rig_path: str | Path,
    datawash_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path | None = None,
    detector: str = "nn",
    nn_model: str | Path | None = None,
    kalibr_image: str = DEFAULT_KALIBR_IMAGE,
    kalibr_detector_processes: int = 0,
    device: str = "auto",
    overwrite: bool = False,
) -> dict:
    detector_name = normalize_detector_name(detector)
    source_path = Path(source).resolve()
    source_type = detect_source_type(source_path)
    rig_file = Path(rig_path).resolve()
    target_file = Path(target_path).resolve()
    rig = load_rig(rig_file)
    models = require_kalibr_models(rig)
    config = load_datawash_config(datawash_path)
    output = Path(output_dir).resolve() if output_dir is not None else default_output_dir(source_path)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"pipeline summary exists: {summary_path}")

    detector_cache = None
    model_path: Path | None = None
    if detector_name == "nn":
        model_path = Path(nn_model).resolve() if nn_model is not None else bundled_nn_model().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"NN-Detector model not found: {model_path}")
    else:
        groups = load_frame_groups(source_path, rig)
        detector_cache = extract_acv_detection_cache(
            groups=groups,
            rig=rig,
            rig_path=rig_file,
            target_path=target_file,
            output_dir=output / ".work" / "acv_detector",
            confidence_threshold=config.detector_confidence,
            docker_image=kalibr_image,
            num_processes=kalibr_detector_processes,
            overwrite=overwrite,
        )

    datawash_summary = run_datawash(
        source=source_path,
        rig=rig,
        config=config,
        model_path=model_path,
        output_dir=output / "datawash",
        input_cache=detector_cache,
        detector=detector_name,
        device=device,
        overwrite=overwrite,
    )
    clean_bag = Path(datawash_summary["clean_ros1_bag"])
    selected_cache = Path(datawash_summary.pop("_selected_detection_cache"))
    full_cache = Path(datawash_summary.pop("_full_detection_cache"))
    kalibr_output = output / "kalibr"
    _run_kalibr(
        detector=detector_name,
        clean_bag=clean_bag,
        selected_cache=selected_cache,
        rig_path=rig_file,
        target_path=target_file,
        output_dir=kalibr_output,
        docker_image=kalibr_image,
    )
    for internal_path in (full_cache, selected_cache, output / ".work"):
        if internal_path.is_dir():
            shutil.rmtree(internal_path)
    summary = {
        "detector": detector_name,
        "requested_device": device,
        "input": str(source_path),
        "input_type": source_type,
        "output": str(output),
        "rig": str(rig_file),
        "models": list(models),
        "target": str(target_file),
        "datawash": datawash_summary,
        "kalibr_output": str(kalibr_output),
        "kalibr_image": str(kalibr_image),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
