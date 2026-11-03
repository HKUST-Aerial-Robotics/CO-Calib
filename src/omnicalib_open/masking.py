"""Estimate static dark lens borders from temporally sampled calibration images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from .rig import load_rig
from .sources import _encode_raw_image, _message_bytes, _read_timestamp_rows, detect_source_type


def sample_indices(count, limit):
    return set(np.linspace(0, count - 1, min(count, limit), dtype=int)) if count else set()


def sampled_images(source, rig, limit):
    """Decode uniformly spaced frames without loading an entire bag into memory."""
    if detect_source_type(source) == 'sequence':
        for camera in rig.cameras:
            rows = sorted(_read_timestamp_rows(source / camera.directory), key=lambda row: row[1])
            for index in sorted(sample_indices(len(rows), limit)):
                yield camera.camera_id, cv2.imread(str(rows[index][2]), cv2.IMREAD_GRAYSCALE)
        return
    with AnyReader([source]) as reader:
        cameras = {camera.topic: camera.camera_id for camera in rig.cameras}
        connections = [c for c in reader.connections if c.topic in cameras]
        missing = set(cameras) - {c.topic for c in connections}
        if missing:
            raise ValueError(f'Missing image topics: {sorted(missing)}')
        counts = {topic: sum(c.msgcount for c in connections if c.topic == topic) for topic in cameras}
        indices = {topic: sample_indices(n, limit) for topic, n in counts.items()}
        seen = dict.fromkeys(cameras, 0)
        for connection, _, raw in reader.messages(connections=connections):
            index = seen[connection.topic]
            seen[connection.topic] += 1
            if index not in indices[connection.topic]:
                continue
            message = reader.deserialize(raw, connection.msgtype)
            if connection.msgtype.endswith('/CompressedImage'):
                encoded = _message_bytes(message.data)
            elif connection.msgtype.endswith('/Image'):
                encoded, _ = _encode_raw_image(message)
            else:
                raise ValueError(f'Not an image topic: {connection.topic}')
            yield cameras[connection.topic], cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_GRAYSCALE)


class BorderEvidence:
    def __init__(self, shape):
        self.count = 0
        self.dark = np.zeros(shape, np.uint32)
        self.flat_dark = np.zeros(shape, np.uint32)
        self.features = np.zeros(shape, np.uint32)
        self.mean = np.zeros(shape, np.float64)

    def add(self, gray, dark_threshold, texture_threshold):
        if gray is None or gray.dtype != np.uint8 or gray.ndim != 2 or gray.shape != self.dark.shape:
            raise ValueError('Each camera must contain decodable uint8 images of a constant size')
        values = gray.astype(np.float32)
        mean = cv2.boxFilter(values, -1, (7, 7))
        variance = np.maximum(cv2.boxFilter(values * values, -1, (7, 7)) - mean * mean, 0)
        dark = gray <= dark_threshold
        self.dark += dark
        self.flat_dark += dark & (variance <= texture_threshold ** 2)
        # Repeated Shi-Tomasi corners protect useful dark texture from masking.
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=0, qualityLevel=0.01, minDistance=5, blockSize=5)
        hits = np.zeros(gray.shape, np.uint8)
        if corners is not None:
            xy = np.rint(corners[:, 0]).astype(int)
            hits[xy[:, 1], xy[:, 0]] = 1
        self.features += cv2.dilate(hits, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        self.mean += values
        self.count += 1

    def finish(self, persistence, feature_rate, margin):
        if self.count < 10:
            raise ValueError('At least 10 sampled frames per camera are required')
        # ponytail: temporal evidence cannot distinguish a stationary dark occluder;
        # record camera/target motion and review overlays before deployment.
        protected = self.features / self.count >= feature_rate
        candidate = ((self.flat_dark / self.count >= persistence) & ~protected).astype(np.uint8)
        _, labels = cv2.connectedComponents(candidate, connectivity=8)
        edges = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        invalid = np.isin(labels, edges[edges != 0]).astype(np.uint8)
        if margin:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            invalid = cv2.dilate(invalid, kernel)
        invalid[protected] = 0
        if invalid.mean() > 0.95:
            raise ValueError('More than 95% of the image would be masked; check exposure and thresholds')
        return ((1 - invalid) * 255).astype(np.uint8)


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write image: {path}')


def generate(args):
    if not 10 <= args.max_frames <= 1000000:
        raise ValueError('--max-frames must be between 10 and 1000000')
    if not 0 <= args.dark_threshold <= 255 or not np.isfinite(args.texture_threshold) or args.texture_threshold < 0:
        raise ValueError('Invalid brightness or texture threshold')
    if not 0 < args.persistence <= 1 or not 0 < args.feature_rate <= 1 or not 0 <= args.margin <= 50:
        raise ValueError('Invalid persistence, feature rate, or margin')
    source = args.input.resolve()
    rig = load_rig(args.rig)
    if any(Path(c.camera_id).name != c.camera_id or c.camera_id in ('.', '..') for c in rig.cameras):
        raise ValueError('Camera IDs must be plain file names')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {output}')
    evidence = {}
    for camera_id, gray in sampled_images(source, rig, args.max_frames):
        if gray is None:
            raise ValueError(f'Cannot decode an image for {camera_id}')
        item = evidence.get(camera_id)
        if item is None:
            item = evidence[camera_id] = BorderEvidence(gray.shape)
        item.add(gray, args.dark_threshold, args.texture_threshold)
        if item.count % 100 == 0:
            print(f'{camera_id}: {item.count} frames', flush=True)
    if set(evidence) != set(rig.camera_ids):
        raise ValueError('Input must contain frames for every configured camera')
    masks = {key: item.finish(args.persistence, args.feature_rate, args.margin) for key, item in evidence.items()}
    output.mkdir(parents=True, exist_ok=True)
    summary = {'input': str(source), 'rig': str(args.rig.resolve()), 'mask_convention': '255=keep, 0=exclude',
               'parameters': {k: v for k, v in vars(args).items() if isinstance(v, (int, float))}, 'cameras': {}}
    panels = []
    for camera in rig.cameras:
        key = camera.camera_id
        item, mask = evidence[key], masks[key]
        write_image(output / f'{key}_mask.png', mask)
        mean = np.rint(item.mean / item.count).astype(np.uint8)
        write_image(output / f'{key}_mean.png', mean)
        write_image(output / f'{key}_dark_frequency.png', np.rint(item.dark / item.count * 255).astype(np.uint8))
        overlay = cv2.cvtColor(mean, cv2.COLOR_GRAY2BGR)
        excluded = mask == 0
        overlay[excluded] = (0.45 * overlay[excluded] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        contours, _ = cv2.findContours(excluded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        cv2.putText(overlay, f'{key}: excluded {excluded.mean():.1%}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        write_image(output / f'{key}_overlay.png', overlay)
        panels.append(cv2.resize(overlay, (384, 360)))
        summary['cameras'][key] = {'topic': camera.topic, 'resolution': [mask.shape[1], mask.shape[0]],
                                  'frames_sampled': item.count, 'excluded_fraction': float(excluded.mean()),
                                  'feature_hits_excluded_fraction': float(item.features[excluded].sum() / max(1, item.features.sum()))}
    columns = min(3, len(panels))
    while len(panels) % columns:
        panels.append(np.zeros_like(panels[0]))
    write_image(output / 'overview.jpg', np.vstack([np.hstack(panels[i:i+columns]) for i in range(0, len(panels), columns)]))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0


def add_arguments(parser):
    parser.add_argument('--input', required=True, type=Path, help='sequence directory, ROS1 bag, or ROS2 bag')
    parser.add_argument('--rig', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='empty output directory')
    parser.add_argument('--max-frames', type=int, default=300, help='uniform samples per camera (default: 300)')
    parser.add_argument('--dark-threshold', type=int, default=45, help='maximum dark grayscale value, 0-255')
    parser.add_argument('--texture-threshold', type=float, default=5, help='maximum local grayscale standard deviation')
    parser.add_argument('--persistence', type=float, default=.95, help='required dark and low-texture frame fraction')
    parser.add_argument('--feature-rate', type=float, default=.01, help='corner occurrence fraction that protects a pixel')
    parser.add_argument('--margin', type=int, default=2, help='extra exclusion margin in pixels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    generate(parser.parse_args())
