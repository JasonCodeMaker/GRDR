#!/usr/bin/env python
from __future__ import annotations

import tempfile
from pathlib import Path

from evaluator import resolve_cached_feature_path


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "PANDA" / "train").mkdir(parents=True)
        (root / "PANDA" / "test").mkdir(parents=True)
        (root / "PANDA" / "train" / "train_vid.npz").write_text("x")
        (root / "PANDA" / "test" / "test_vid.npz").write_text("x")

        assert resolve_cached_feature_path(str(root), "PANDA", "train_vid") == root / "PANDA" / "train" / "train_vid.npz"
        assert resolve_cached_feature_path(str(root / "PANDA"), "PANDA", "test_vid") == root / "PANDA" / "test" / "test_vid.npz"
        assert resolve_cached_feature_path(str(root), "MSRVTT", "missing") == root / "missing.npz"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
