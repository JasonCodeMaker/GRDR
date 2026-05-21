import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


class NpyMemmapMapping(Mapping):
    """Read-only Mapping over a .npy array plus a JSON key-to-row index."""

    def __init__(self, npy_path, idx_path):
        self.npy_path = Path(npy_path)
        self.idx_path = Path(idx_path)
        self._arr = np.load(self.npy_path, mmap_mode='r')
        with self.idx_path.open('r') as f:
            self._idx = json.load(f)

    def __getitem__(self, key):
        return self._arr[self._idx[key]]

    def __iter__(self):
        return iter(self._idx)

    def __len__(self):
        return len(self._idx)

    def __contains__(self, key):
        return key in self._idx


class MappingUnion(Mapping):
    """Read-only union where native keys take precedence over overlay keys."""

    def __init__(self, native, overlay, assume_disjoint=True):
        self._native = native
        self._overlay = overlay
        self._assume_disjoint = assume_disjoint
        if assume_disjoint:
            self._len = len(native) + len(overlay)
        else:
            self._len = len(native) + sum(1 for k in overlay if k not in native)

    def __getitem__(self, key):
        if key in self._native:
            return self._native[key]
        return self._overlay[key]

    def __iter__(self):
        yield from self._native
        if self._assume_disjoint:
            yield from self._overlay
        else:
            yield from (k for k in self._overlay if k not in self._native)

    def __len__(self):
        return self._len

    def __contains__(self, key):
        return key in self._native or key in self._overlay
