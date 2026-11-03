from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import DetectionRecord


CACHE_FORMAT = "omnicalib.detection-cache"
LEGACY_CACHE_FORMAT = "omnicalib.nncache"
CACHE_VERSION = 1


def normalize_detector_name(detector: str) -> str:
    name = str(detector).strip().lower().replace("_", "-")
    aliases = {
        "nn-detector": "nn",
        "nndetector": "nn",
        "acv-detector": "acv",
        "acvdetector": "acv",
        "kalibr": "acv",
    }
    name = aliases.get(name, name)
    if name not in {"nn", "acv"}:
        raise ValueError("detector must be NN-Detector or ACV-Detector")
    return name


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_cache(
    path: str | Path,
    records: Iterable[DetectionRecord],
    *,
    model_path: str | Path | None,
    confidence_threshold: float,
    camera_ids: Iterable[str],
    detector: str = "nn",
    subpixel: dict | None = None,
    overwrite: bool = False,
) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    arrays_path = output / "observations.npz"
    if (manifest_path.exists() or arrays_path.exists()) and not overwrite:
        raise FileExistsError(f"cache exists: {output}")

    rows = list(records)
    if rows:
        point_counts = {np.asarray(row.points_xy).reshape(-1, 2).shape[0] for row in rows}
        if len(point_counts) != 1:
            raise ValueError("All cache records must use the same ordered target point count")
        points = np.stack([np.asarray(row.points_xy, dtype=np.float32).reshape(-1, 2) for row in rows])
        confidence = np.stack([np.asarray(row.confidence, dtype=np.float32).reshape(-1) for row in rows])
    else:
        points = np.zeros((0, 0, 2), dtype=np.float32)
        confidence = np.zeros((0, 0), dtype=np.float32)

    np.savez_compressed(
        arrays_path,
        camera_id=np.asarray([row.camera_id for row in rows], dtype=np.str_),
        source_index=np.asarray([row.source_index for row in rows], dtype=np.int64),
        timestamp_ns=np.asarray([row.timestamp_ns for row in rows], dtype=np.int64),
        image_height=np.asarray([row.image_height for row in rows], dtype=np.int32),
        image_width=np.asarray([row.image_width for row in rows], dtype=np.int32),
        xy_px=points,
        confidence=confidence,
    )
    detector_name = normalize_detector_name(detector)
    if detector_name == "nn" and model_path is None:
        raise ValueError("NN-Detector cache requires model_path")
    manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "detector": detector_name,
        "model_sha256": sha256_file(model_path) if model_path is not None else None,
        "confidence_threshold": float(confidence_threshold),
        "camera_ids": list(camera_ids),
        "corner_order": "tag-major; model-native BR,BL,TL,TR",
        "record_count": len(rows),
        "arrays": "observations.npz",
        "subpixel": subpixel,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_cache(
    path: str | Path,
    *,
    expected_model: str | Path | None = None,
    expected_detector: str | None = None,
) -> tuple[dict, list[DetectionRecord]]:
    cache_path = Path(path).resolve()
    manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") not in {CACHE_FORMAT, LEGACY_CACHE_FORMAT} or int(manifest.get("version", -1)) != CACHE_VERSION:
        raise ValueError(f"Unsupported detection cache: {cache_path}")
    detector = normalize_detector_name(str(manifest.get("detector", "nn")))
    manifest["detector"] = detector
    if expected_detector is not None and detector != normalize_detector_name(expected_detector):
        raise ValueError(f"Detection cache uses {detector}, expected {expected_detector}")
    if expected_model is not None and manifest.get("model_sha256") != sha256_file(expected_model):
        raise ValueError("NN-Detector cache model hash does not match the requested model")
    with np.load(cache_path / str(manifest["arrays"]), allow_pickle=False) as archive:
        # NPZ members are decompressed on every access; materialize each once.
        arrays = {key: archive[key] for key in archive.files}
        rows = [
            DetectionRecord(
                camera_id=str(arrays["camera_id"][index]),
                source_index=int(arrays["source_index"][index]),
                timestamp_ns=int(arrays["timestamp_ns"][index]),
                image_height=int(arrays["image_height"][index]),
                image_width=int(arrays["image_width"][index]),
                points_xy=np.asarray(arrays["xy_px"][index], dtype=np.float32).copy(),
                confidence=np.asarray(arrays["confidence"][index], dtype=np.float32).copy(),
            )
            for index in range(len(arrays["camera_id"]))
        ]
    if len(rows) != int(manifest.get("record_count", -1)):
        raise ValueError("Detection cache record count is inconsistent")
    return manifest, rows
