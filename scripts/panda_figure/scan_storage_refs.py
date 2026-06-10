"""Scan the repo for GRDR/ANN storage-calculation references and flag drift from
the Storage Accounting SSOT (storage_accounting_ssot.md / storage_ssot.py).

Reusable workflow: re-run after any SSOT change. Buckets each hit so a human (or a
delegated agent) can triage:
  STALE   - old GRDR storage signatures (275 MB decoder, 21.72/~22 B/vid, 319 MB, D_decoder/T_trie)
  COMPUTE - code that computes storage (must import storage_ssot)
  REPORT  - prose/tables that report per-video or total storage (must cite SSOT values)
  SSOT_OK - already references the SSOT (informational)

Usage:  python scripts/panda_figure/scan_storage_refs.py [--stale-only]
Exit 1 if any STALE hit exists outside the allowlist.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP_DIRS = {".git", "node_modules", "graphify-out", ".worktrees", "wandb", "__pycache__",
             "video_features_cache", "checkpoints", "feature_data", ".conda-env",
             "brainstorm"}  # brainstorm/ docs are frozen-on-promotion history, not authoritative

# A STALE hit on a line ALSO matching SUPPRESS is intentional registration/history (a registered
# hypothesis, a footer history note, or the SSOT rule that names the old number to forbid it).
SUPPRESS = re.compile(r'data-hypothesis-restated|data-invariant="hypothesis"|plan-invariant'
                      r'|hypothesis:|^\s*objective:|^\s*direction:|^\s*lastDecision:'
                      r'|data-field="objective|data-field="direction"'
                      r'|was decoder-dominated|the old ~?275|migrated to project SSOT'
                      r'|do NOT count a fixed decoder')
EXTS = (".py", ".html", ".md", ".ipynb", ".js", ".json", ".csv")
MAX_BYTES = 1_000_000  # storage reporting lives in small docs/code; skip giant data/candidate files
# Files allowed to carry the OLD numbers (the SSOT itself + this scanner + history notes).
ALLOW = ("storage_accounting_ssot.md", "scan_storage_refs.py", "storage_ssot.py")

PATTERNS = [
    ("STALE", re.compile(r"275674345|275\s?MB|275&nbsp;MB|~?320\s?MB|319\.2|\b21\.72\b|D_decoder|T_trie|decoder-dominated|T\+M\+D")),
    ("COMPUTE", re.compile(r"storage_sim|\.write_index|serialize_index|bytes_per_video|getsize\([^)]*(?:ckpt|decoder|model)")),
    ("REPORT", re.compile(r"\bB/?vid\b|bytes/?video|per-video.*(?:byte|storage)|storage_data|no[- ]side[- ]table|index (?:size|footprint)")),
    ("SSOT_OK", re.compile(r"storage_accounting_ssot|Storage Accounting SSOT|32\s?(?:&nbsp;)?B/?vid")),
]


def scan():
    hits = {cat: [] for cat, _ in PATTERNS}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if not fn.endswith(EXTS) or any(a in fn for a in ALLOW):
                continue
            path = os.path.join(root, fn)
            try:
                if os.path.getsize(path) > MAX_BYTES:
                    continue
            except OSError:
                continue
            rel = os.path.relpath(path, REPO)
            try:
                with open(path, errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        for cat, rx in PATTERNS:
                            if rx.search(line):
                                if cat == "STALE" and SUPPRESS.search(line):
                                    continue
                                hits[cat].append((rel, i, line.strip()[:160]))
            except OSError:
                continue
    return hits


def main():
    stale_only = "--stale-only" in sys.argv
    hits = scan()
    order = ["STALE", "COMPUTE", "REPORT", "SSOT_OK"]
    for cat in order:
        if stale_only and cat != "STALE":
            continue
        rows = hits[cat]
        print(f"\n===== {cat} ({len(rows)} hits, {len({r[0] for r in rows})} files) =====")
        for rel, i, snip in rows:
            print(f"{rel}:{i}: {snip}")
    print(f"\nSUMMARY: " + ", ".join(f"{c}={len(hits[c])}" for c in order))
    sys.exit(1 if hits["STALE"] else 0)


if __name__ == "__main__":
    main()
