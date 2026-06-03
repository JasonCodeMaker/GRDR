#!/usr/bin/env python
"""Build the 8 per-(dataset, setting) latency-subset manifests for Pass B.

IMPORTANT: must be run with CWD = GRDR repo root. test_perquery.load_test_queries
uses relative paths like `reranker/xpool/data/<DS>/...` to locate the canonical
test split CSV/JSON files; running from elsewhere will hit FileNotFoundError.

Example:
  cd /home/uqzzha35/Project/SemanticID/GRDR && \\
    python scripts/latency_recall_figure/lib/build_latency_subset.py \\
      --output_dir output/evaluation_results/figures_panda/manifests/latency
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

import numpy as np

# Reuse X-Pool's canonical query loader so the manifest's query_ids are
# exactly the ids every Pass-A and Pass-B run sees.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
XPOOL_DIR = os.path.join(REPO_ROOT, 'reranker', 'xpool')
sys.path.insert(0, XPOOL_DIR)
from test_perquery import load_test_queries  # noqa: E402

DATASETS = ['MSRVTT', 'ACTNET', 'DIDEMO', 'LSMDC']
SETTINGS = [1, 2]
ACTNET_S2_MAX_QUERIES = 1000  # matches Pass-A convention


class _Cfg:
    """Minimal duck-typed config for load_test_queries."""
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name


def _sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _build_one(dataset, setting, warmup_n, subset_n, seed):
    cfg = _Cfg(dataset)
    queries = load_test_queries(cfg)
    if dataset == 'ACTNET' and setting == 2:
        queries = queries[:ACTNET_S2_MAX_QUERIES]
    # qid = canonical video_id. Test splits have one query per video for the
    # 4 datasets in scope, so this is unique. Each baseline filters its loader
    # by video_id directly, avoiding loader-order dependence.
    qids = [vid for (_, vid) in queries]
    if len(qids) != len(set(qids)):
        # Disambiguate the rare duplicate-video case by suffixing the loader index.
        seen = {}
        unique = []
        for i, vid in enumerate(qids):
            seen[vid] = seen.get(vid, 0) + 1
            unique.append(f"{vid}#dup{seen[vid]}" if seen[vid] > 1 else vid)
        qids = unique
    qids = sorted(qids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(qids))
    shuffled = [qids[i] for i in perm]
    n_need = warmup_n + subset_n
    if len(shuffled) < n_need:
        raise ValueError(
            f"{dataset} t{setting}: only {len(shuffled)} queries, need {n_need}"
        )
    warmup = shuffled[:warmup_n]
    timed = shuffled[warmup_n:warmup_n + subset_n]
    payload = {
        'metadata': {
            'dataset': dataset,
            'setting': setting,
            'subset_seed': seed,
            'subset_n_target': subset_n,
            'warmup_n': warmup_n,
            'source_query_count': len(queries),
            'build_timestamp': _dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
        'warmup_query_ids': warmup,
        'timed_query_ids': timed,
    }
    payload['metadata']['content_sha256'] = _sha256(
        {'warmup_query_ids': warmup, 'timed_query_ids': timed}
    )
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True)
    p.add_argument('--warmup_n', type=int, default=10)
    p.add_argument('--subset_n', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--settings', nargs='+', type=int, default=SETTINGS)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for ds in args.datasets:
        for st in args.settings:
            out_path = os.path.join(args.output_dir, f'latency_subset_{ds}_t{st}.json')
            payload = _build_one(ds, st, args.warmup_n, args.subset_n, args.seed)
            with open(out_path, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"wrote {out_path}  warmup={len(payload['warmup_query_ids'])}"
                  f"  timed={len(payload['timed_query_ids'])}")


if __name__ == '__main__':
    main()
