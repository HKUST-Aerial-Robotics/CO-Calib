"""Run with PYTHONPATH=src python tests/test_cache.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from numpy.lib.npyio import NpzFile
from omnicalib_open.cache import read_cache, write_cache
from omnicalib_open.models import DetectionRecord

with TemporaryDirectory() as folder:
    records = [DetectionRecord(camera_id='cam0', source_index=i, timestamp_ns=100+i,
               image_height=60, image_width=80, points_xy=np.array([[i, 2], [3, 4]], np.float32),
               confidence=np.array([.99, .9], np.float32)) for i in range(12)]
    dest = write_cache(Path(folder)/'cache', records, model_path=None,
                       confidence_threshold=.9, camera_ids=['cam0'], detector='acv')
    accesses = []
    getitem = NpzFile.__getitem__
    def count_access(archive, key):
        accesses.append(key)
        return getitem(archive, key)
    with patch.object(NpzFile, '__getitem__', count_access):
        _, loaded = read_cache(dest, expected_detector='acv')
    assert len(accesses) == len(set(accesses)), 'Each NPZ member must be decompressed only once'
    assert len(loaded) == len(records)
    for a, b in zip(records, loaded):
        assert a.source_index == b.source_index and a.timestamp_ns == b.timestamp_ns
        assert np.array_equal(a.points_xy, b.points_xy) and np.array_equal(a.confidence, b.confidence)
    try:
        read_cache(dest, expected_detector='nn')
    except ValueError:
        pass
    else:
        raise AssertionError('Detector mismatch must be rejected')
print('Cache roundtrip and single-decompression checks passed.')
