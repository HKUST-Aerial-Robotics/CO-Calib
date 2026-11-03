from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .bag_writer import write_clean_ros1_bag
from .cache import normalize_detector_name, read_cache, write_cache
from .models import (
    CameraMask,
    CameraObservation,
    CandidateGroup,
    DatawashConfig,
    DetectionRecord,
    FrameGroup,
    ResizePlan,
    RigSpec,
    SelectedGroup,
)
from .nn_detector import NNDetector
from .preprocess import compute_resize_plan, remap_points_to_original, resize_image
from .quality import RadialSpanNormalizer, build_camera_mask, evaluate_observation
from .selection import select_groups
from .sources import all_records, decode_frame, load_frame_groups
from .subpixel import SUBPIXEL, refine_points


_DETECTOR_MIN_SIDE = 720
_DETECTOR_MAX_SIDE = 1280
_GRID_CELL_PX = 10
_MASK_GRAY_THRESHOLD = 5
_MASK_CELL_MIN_FRACTION = 0.20
_MASK_EDGE_RATIO = 0.0


def detect_groups(
    groups: Sequence[FrameGroup],
    *,
    detector: NNDetector,
    config: DatawashConfig,
) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    source_records = sorted(all_records(groups), key=lambda item: (item.timestamp_ns, item.camera_id))
    for index, frame in enumerate(source_records, start=1):
        image = decode_frame(frame)
        height, width = image.shape[:2]
        plan = compute_resize_plan(height, width, _DETECTOR_MIN_SIDE, _DETECTOR_MAX_SIDE)
        inference_image = resize_image(image, plan)
        output = detector.detect(inference_image)
        points = remap_points_to_original(np.asarray(output.points_xy, dtype=np.float32), plan)
        points = refine_points(image, points)
        records.append(
            DetectionRecord(
                camera_id=frame.camera_id,
                source_index=frame.source_index,
                timestamp_ns=frame.timestamp_ns,
                image_height=height,
                image_width=width,
                points_xy=points,
                confidence=np.asarray(output.confidence, dtype=np.float32).reshape(-1),
            )
        )
        if index == 1 or index % 100 == 0 or index == len(source_records):
            print(json.dumps({"phase": "nn_detect", "done": index, "total": len(source_records)}), flush=True)
    return records


def _build_masks(
    groups: Sequence[FrameGroup], rig: RigSpec
) -> tuple[dict[str, CameraMask], dict[str, RadialSpanNormalizer]]:
    first_by_camera = {}
    for group in groups:
        for camera_id, frame in group.frames.items():
            first_by_camera.setdefault(camera_id, frame)
    missing = sorted(set(rig.camera_ids) - set(first_by_camera))
    if missing:
        raise ValueError(f"Input contains no frames for: {', '.join(missing)}")
    masks: dict[str, CameraMask] = {}
    radial_normalizers: dict[str, RadialSpanNormalizer] = {}
    for camera_id in rig.camera_ids:
        image = decode_frame(first_by_camera[camera_id])
        masks[camera_id] = build_camera_mask(
            camera_id,
            image,
            grid_cell_px=_GRID_CELL_PX,
            gray_thr=_MASK_GRAY_THRESHOLD,
            cell_min_frac=_MASK_CELL_MIN_FRACTION,
            edge_ratio=_MASK_EDGE_RATIO,
        )
        radial_normalizers[camera_id] = RadialSpanNormalizer.from_image(
            image,
            gray_threshold=_MASK_GRAY_THRESHOLD,
        )
    return masks, radial_normalizers


def _missing_observation(camera_id: str, timestamp_ns: int, mask: CameraMask) -> CameraObservation:
    resize = ResizePlan(
        mask.image_height,
        mask.image_width,
        mask.image_height,
        mask.image_width,
        1.0,
        1.0,
        1.0,
    )
    return CameraObservation(
        camera_id=camera_id,
        image_path=None,
        timestamp_ns=timestamp_ns,
        resize=resize,
        mask=mask,
        valid_points_xy=np.zeros((0, 2), dtype=np.float32),
        coverage_mask=0,
        center_hit=False,
        usable=False,
        detected_points=0,
        projective_quality=0.0,
        projective_local_shape=0.0,
        projective_area_uniformity=0.0,
        projective_jacobian_isotropy=0.0,
        projective_area_gradient=0.0,
        projective_homography_rms_px=float("inf"),
        projective_valid_tags=0,
        diagonal_ratio=0.0,
        diagonal_valid_tags=0,
    )


def build_candidates(
    groups: Sequence[FrameGroup],
    records: Sequence[DetectionRecord],
    *,
    masks: dict[str, CameraMask],
    radial_normalizers: dict[str, RadialSpanNormalizer],
    config: DatawashConfig,
) -> list[CandidateGroup]:
    detections = {(row.camera_id, row.source_index, row.timestamp_ns): row for row in records}
    candidates = []
    for group in groups:
        observations = {}
        for camera_id, mask in masks.items():
            frame = group.frames.get(camera_id)
            if frame is None:
                observations[camera_id] = _missing_observation(camera_id, group.timestamp_ns, mask)
                continue
            key = (camera_id, frame.source_index, frame.timestamp_ns)
            if key not in detections:
                raise ValueError(f"Detection cache is missing {camera_id} frame {frame.source_index}")
            detection = detections[key]
            plan = ResizePlan(
                detection.image_height,
                detection.image_width,
                detection.image_height,
                detection.image_width,
                1.0,
                1.0,
                1.0,
            )
            observations[camera_id] = evaluate_observation(
                camera_id=camera_id,
                image_path=frame.source_path,
                timestamp_ns=frame.timestamp_ns,
                resize=plan,
                mask=mask,
                points_xy=np.asarray(detection.points_xy, dtype=np.float32),
                confidence=np.asarray(detection.confidence, dtype=np.float32),
                conf_thr=config.detector_confidence,
                radial_normalizer=radial_normalizers[camera_id],
            )
        candidates.append(CandidateGroup(group.frame_id, group.timestamp_ns, observations))
    return candidates


def _selected_detection_records(
    selected: Sequence[SelectedGroup], groups: Sequence[FrameGroup], records: Sequence[DetectionRecord]
) -> list[DetectionRecord]:
    group_by_id = {group.frame_id: group for group in groups}
    keys = set()
    for selected_group in selected:
        source_group = group_by_id[selected_group.frame_id]
        for camera_id in selected_group.active_cams:
            frame = source_group.frames[camera_id]
            keys.add((camera_id, frame.source_index, frame.timestamp_ns))
    return [row for row in records if (row.camera_id, row.source_index, row.timestamp_ns) in keys]


def _write_roles(
    path: Path,
    selected: Sequence[SelectedGroup],
    role_by_key: dict[tuple[int, tuple[str, ...]], str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("frame_id", "timestamp_ns", "active_cameras", "role"))
        writer.writeheader()
        for group in selected:
            active = tuple(sorted(group.active_cams))
            writer.writerow(
                {
                    "frame_id": group.frame_id,
                    "timestamp_ns": group.timestamp_ns,
                    "active_cameras": ";".join(active),
                    "role": role_by_key.get((group.frame_id, active), ""),
                }
            )


def run_datawash(
    *,
    source: str | Path,
    rig: RigSpec,
    config: DatawashConfig,
    model_path: str | Path | None,
    output_dir: str | Path,
    input_cache: str | Path | None = None,
    detector: str = "nn",
    device: str = "auto",
    overwrite: bool = False,
) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        output / "detections_full.detcache" / "manifest.json",
        output / "detections_selected.detcache" / "manifest.json",
        output / "calibration_clean.bag",
        output / "selected_roles.csv",
        output / "summary.json",
    )
    if not overwrite:
        existing = [path for path in expected_outputs if path.exists()]
        if existing:
            raise FileExistsError(f"output already contains Datawash artifacts: {existing[0]}")
    groups = load_frame_groups(source, rig)
    detector_name = normalize_detector_name(detector)
    resolved_model = Path(model_path).resolve() if model_path is not None else None
    if detector_name == "nn" and resolved_model is None:
        raise ValueError("NN-Detector requires model_path")
    if input_cache is None:
        if detector_name != "nn":
            raise ValueError("ACV-Detector mode requires an extracted input_cache")
        nn_detector = NNDetector(resolved_model, device=device)
        records = detect_groups(groups, detector=nn_detector, config=config)
        execution_providers = list(nn_detector.providers)
    else:
        manifest, records = read_cache(
            input_cache,
            expected_model=resolved_model if detector_name == "nn" else None,
            expected_detector=detector_name,
        )
        cached_threshold = float(manifest["confidence_threshold"])
        if detector_name == "nn" and manifest.get("subpixel") != SUBPIXEL:
            raise ValueError("NN-Detector cache must contain the current original-image subpixel refinement")
        if abs(cached_threshold - config.detector_confidence) > 1e-9:
            raise ValueError(
                f"Cache confidence threshold {cached_threshold} does not match config {config.detector_confidence}"
            )
        execution_providers = []

    full_cache = write_cache(
        output / "detections_full.detcache",
        records,
        model_path=resolved_model,
        confidence_threshold=config.detector_confidence,
        camera_ids=rig.camera_ids,
        detector=detector_name,
        subpixel=SUBPIXEL if detector_name == "nn" else None,
        overwrite=overwrite,
    )
    masks, radial_normalizers = _build_masks(groups, rig)
    candidates = build_candidates(
        groups,
        records,
        masks=masks,
        radial_normalizers=radial_normalizers,
        config=config,
    )
    selected, selection_summary, roles = select_groups(
        candidates,
        masks,
        config.selection,
        sample_min_detection_points=config.sample_min_detection_points,
    )
    selected_records = _selected_detection_records(selected, groups, records)
    selected_cache = write_cache(
        output / "detections_selected.detcache",
        selected_records,
        model_path=resolved_model,
        confidence_threshold=config.detector_confidence,
        camera_ids=rig.camera_ids,
        detector=detector_name,
        subpixel=SUBPIXEL if detector_name == "nn" else None,
        overwrite=overwrite,
    )
    clean_bag = write_clean_ros1_bag(
        output / "calibration_clean.bag",
        groups=groups,
        selected=selected,
        rig=rig,
        overwrite=overwrite,
    )
    roles_path = output / "selected_roles.csv"
    if roles_path.exists() and not overwrite:
        raise FileExistsError(f"roles manifest exists: {roles_path}")
    _write_roles(roles_path, selected, roles)
    summary = {
        "input": str(Path(source).resolve()),
        "camera_ids": list(rig.camera_ids),
        "detector": detector_name,
        "subpixel": dict(SUBPIXEL) if detector_name == "nn" else None,
        "execution_providers": execution_providers,
        "source_groups": len(groups),
        "source_images": len(all_records(groups)),
        "clean_ros1_bag": str(clean_bag),
        "roles": str(roles_path),
        "selection": selection_summary,
    }
    summary_path = output / "summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"summary exists: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **summary,
        "_full_detection_cache": str(full_cache),
        "_selected_detection_cache": str(selected_cache),
    }
