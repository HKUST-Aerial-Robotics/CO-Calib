"""Run with PYTHONPATH=src python tests/test_masking.py."""
import numpy as np
from omnicalib_open.masking import BorderEvidence, sample_indices

item = BorderEvidence((100, 120))
for index in range(20):
    gray = np.full((100, 120), 160, np.uint8)
    gray[:, :20] = 20
    gray[35:65, 55:85] = 20  # Interior darkness must not become a lens border.
    gray[30:70, :20] = 160 if index < 4 else 20  # Intermittently visible border area.
    item.add(gray, 45, 5)
mask = item.finish(.95, .01, 0)
assert mask[10, 5] == 0
assert mask[50, 5] == 255
assert mask[50, 70] == 255
assert mask[20, 100] == 255
assert set(np.unique(mask)) == {0, 255}
item.features[10, 5] = 2
assert item.finish(.95, .01, 0)[10, 5] == 255
assert sample_indices(100, 3) == {0, 49, 99}
assert sample_indices(2, 300) == {0, 1}
try:
    BorderEvidence((10, 10)).finish(.95, .01, 0)
except ValueError:
    pass
else:
    raise AssertionError('Insufficient samples must be rejected')
black = BorderEvidence((50, 50))
for _ in range(10):
    black.add(np.zeros((50, 50), np.uint8), 45, 5)
try:
    black.finish(.95, .01, 0)
except ValueError:
    pass
else:
    raise AssertionError('All-dark recordings must be rejected')
print('Dark-border, interior, intermittent visibility, feature protection, and input checks passed.')
