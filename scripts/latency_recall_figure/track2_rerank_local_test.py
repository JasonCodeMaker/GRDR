#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).with_name("track2_rerank_local.sh")


def main() -> int:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "test_perquery.py" in text, "Panda Setting 2 rerank must use candidate-subset test_perquery.py"
    assert "--index_safe_candidates" in text, "candidate rerank must not key duplicate queries by text"
    assert "--checkpoint" in text, "test_perquery.py expects --checkpoint, not --eval_checkpoint"
    assert "--cache_dir \"${CACHE_PARENT}/PANDA\"" in text, "Panda cache root must expose train/test split dirs"
    assert "--max_queries \"${MAX_QUERIES}\"" in text, "smoke runs need a bounded query count"
    assert 'k.replace("R", "R@")' not in text, "MedR/MeanR must not be normalized to MedR@/MeanR@"

    perquery_pos = text.index("test_perquery.py")
    legacy_branch_pos = text.index("\nelse\n", perquery_pos)
    expanded_pool_pos = text.find("--expanded_pool", perquery_pos, legacy_branch_pos)
    assert expanded_pool_pos == -1, "Panda candidate rerank branch must not pass --expanded_pool"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
