"""Unified per-query latency reporter for Stage-1 + Stage-2 retrieval.

Takes one or more candidate JSON files (produced by eval_ann.py with
--per_query_timing, or by GRDR's trainer/evaluator.py if it exposes the
same `stage1_timing_ms` field) and reports:

  T_text_encode | T_search | T_rerank | total

per query, then aggregated mean/std. Stage-2 (X-Pool rerank) timings are
obtained by reusing reranker/xpool/utils/inference_sim.InferenceLatencySimulator
to time the K candidates from the JSON for each query — no re-implementation.
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

# Make XPool modules importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'reranker', 'xpool'))

from utils.inference_sim import (  # noqa: E402
    InferenceLatencySimulator, SimulatorConfig,
)


DATASET_NAME_MAP = {
    'msrvtt': 'MSRVTT', 'actnet': 'ACTNET',
    'didemo': 'DIDEMO', 'lsmdc': 'LSMDC',
}
CHECKPOINTS = {
    'msrvtt': 'reranker/xpool/ckpt/msrvtt9k_model_best.pth',
    'actnet': 'reranker/xpool/ckpt/actnet_model_best.pth',
    'didemo': 'reranker/xpool/ckpt/didemo_model_best.pth',
    'lsmdc':  'reranker/xpool/ckpt/lsmdc_model_best.pth',
}


def parse_args():
    p = argparse.ArgumentParser(description="Unified Stage-1+Stage-2 latency reporter")
    p.add_argument('--candidate_files', nargs='+', required=True,
                   help='Candidate JSONs to time (e.g., ANN-Flat output and '
                        'GRDR Stage-1 output, both with the same K).')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Optional labels; one per --candidate_files.')
    p.add_argument('--max_queries', type=int, default=100,
                   help='Cap on queries to time per file (for speed).')
    p.add_argument('--num_warmup', type=int, default=5)
    p.add_argument('--num_runs_per_query', type=int, default=3)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--feature_type', type=str, default='frame',
                   choices=['frame', 'video'],
                   help='"frame" matches XPool reranker (frame_embeds + pool_frames).')
    p.add_argument('--pooling_type', type=str, default='attention')
    p.add_argument('--video_batch_size', type=int, default=5000)
    p.add_argument('--cache_root', type=str,
                   default='reranker/xpool/video_features_cache/Xpool')
    p.add_argument('--checkpoint_override', type=str, default=None,
                   help='If set, use this XPool checkpoint for ALL files. '
                        'Otherwise per-file checkpoint is inferred from dataset.')
    p.add_argument('--output', type=str,
                   default='output/reranker/latency_unified.json')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_candidate_file(path):
    with open(path) as f:
        return json.load(f)


def build_simulator(dataset, args, checkpoint):
    """Construct InferenceLatencySimulator (cache + XPool model)."""
    from config.all_config import AllConfig
    from model.model_factory import ModelFactory
    from transformers import CLIPTokenizer

    cache_dir = os.path.join(args.cache_root, DATASET_NAME_MAP[dataset])
    sim_config = SimulatorConfig(
        cache_dir=cache_dir,
        corpus_sizes=[1],  # unused; we drive _run_single_query directly
        num_runs_per_query=args.num_runs_per_query,
        num_warmup_queries=args.num_warmup,
        num_test_queries=args.max_queries,
        seed=args.seed,
        pooling_type=args.pooling_type,
        video_batch_size=args.video_batch_size,
        feature_type=args.feature_type,
    )

    model_config = AllConfig()
    model_config.pooling_type = args.pooling_type
    model = ModelFactory.get_model(model_config)
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state)
    else:
        print(f"WARN: checkpoint missing: {checkpoint} (using random weights)")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return InferenceLatencySimulator(model, tokenizer, sim_config)


def time_file(path, args, label):
    """Time a single candidate JSON: pull Stage-1 from metadata, measure Stage-2."""
    blob = load_candidate_file(path)
    meta = blob['metadata']
    dataset = meta['dataset']
    k = meta['num_candidates']
    results = blob['results']

    print(f"\n{'='*70}")
    print(f"[{label}] {path}")
    print(f"  dataset={dataset} setting={meta['setting']} index={meta.get('index_type')} K={k} pool={meta.get('pool_size')}")
    print(f"  pre-recorded stage1: {'yes' if any('stage1_timing_ms' in r for r in results[:5]) else 'no'}")

    checkpoint = args.checkpoint_override or CHECKPOINTS[dataset]
    simulator = build_simulator(dataset, args, checkpoint)

    # Warmup with the first few queries against their own candidate sets.
    print(f"  Stage-2 warmup ({args.num_warmup} queries)")
    for i in range(min(args.num_warmup, len(results))):
        simulator._run_single_query(results[i]['query_text'],
                                    results[i]['candidates'])

    # Per-query Stage-2 timing on the K candidates from each row.
    n = min(args.max_queries, len(results))
    rows = []
    for i in tqdm(range(n), desc=f"Timing {label}"):
        row = results[i]
        cands = row['candidates']
        if not cands:
            continue
        per_run_total, per_run_enc, per_run_load, per_run_pool, per_run_sim = (
            [], [], [], [], [])
        for _ in range(args.num_runs_per_query):
            tr = simulator._run_single_query(row['query_text'], cands)
            per_run_total.append(tr.total * 1000.0)
            per_run_enc.append(tr.query_encode * 1000.0)
            per_run_load.append(tr.video_load * 1000.0)
            per_run_pool.append(tr.frame_pooling * 1000.0)
            per_run_sim.append(tr.similarity_compute * 1000.0)

        s1 = row.get('stage1_timing_ms', {})
        rows.append({
            't_stage1_encode_ms': s1.get('text_encode'),
            't_stage1_search_ms': s1.get('search'),
            't_stage2_rerank_ms_mean': float(np.mean(per_run_total)),
            't_stage2_rerank_ms_std': float(np.std(per_run_total)),
            't_stage2_breakdown_ms': {
                'text_encode_mean': float(np.mean(per_run_enc)),
                'video_load_mean': float(np.mean(per_run_load)),
                'frame_pool_mean': float(np.mean(per_run_pool)),
                'similarity_mean': float(np.mean(per_run_sim)),
            },
            'k': len(cands),
        })

    # Aggregate.
    enc1 = np.array([r['t_stage1_encode_ms'] for r in rows
                     if r['t_stage1_encode_ms'] is not None])
    sea1 = np.array([r['t_stage1_search_ms'] for r in rows
                     if r['t_stage1_search_ms'] is not None])
    rer = np.array([r['t_stage2_rerank_ms_mean'] for r in rows])
    has_s1 = enc1.size > 0 and sea1.size > 0
    summary = {
        'label': label,
        'file': path,
        'dataset': dataset,
        'setting': meta['setting'],
        'index_type': meta.get('index_type'),
        'k': k,
        'pool_size': meta.get('pool_size'),
        'num_queries_timed': len(rows),
        'num_runs_per_query': args.num_runs_per_query,
        't_text_encode_ms': {
            'mean': float(enc1.mean()) if has_s1 else None,
            'std':  float(enc1.std())  if has_s1 else None,
        },
        't_search_ms': {
            'mean': float(sea1.mean()) if has_s1 else None,
            'std':  float(sea1.std())  if has_s1 else None,
        },
        't_rerank_ms': {
            'mean': float(rer.mean()),
            'std':  float(rer.std()),
        },
        't_total_ms': {
            'mean': float((enc1.mean() if has_s1 else 0)
                          + (sea1.mean() if has_s1 else 0)
                          + rer.mean()),
        },
        'per_query_rows': rows,
    }
    return summary


def print_table(summaries):
    print("\n" + "="*110)
    print(f"{'Label':<28} {'DS':<7} {'set':<4} {'idx':<6} {'K':<5} "
          f"{'Encode(ms)':<14} {'Search(ms)':<14} {'Rerank(ms)':<16} {'Total(ms)':<12}")
    print("-"*110)
    for s in summaries:
        e = s['t_text_encode_ms']['mean']
        se = s['t_search_ms']['mean']
        r = s['t_rerank_ms']['mean']
        t = s['t_total_ms']['mean']
        e_str  = f"{e:>6.2f}" if e  is not None else "  N/A "
        se_str = f"{se:>6.2f}" if se is not None else "  N/A "
        print(f"{s['label']:<28} {s['dataset']:<7} t{s['setting']:<3} "
              f"{(s['index_type'] or '-'):<6} {s['k']:<5} "
              f"{e_str:<14} {se_str:<14} "
              f"{r:>6.2f} +/- {s['t_rerank_ms']['std']:<5.2f} "
              f"{t:>8.2f}")
    print("="*110)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    if args.labels and len(args.labels) != len(args.candidate_files):
        raise SystemExit("--labels length must match --candidate_files length")

    summaries = []
    for i, path in enumerate(args.candidate_files):
        label = args.labels[i] if args.labels else os.path.basename(path)
        summaries.append(time_file(path, args, label))
        # Free GPU between files.
        torch.cuda.empty_cache()

    print_table(summaries)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'summaries': summaries}, f, indent=2)
    print(f"\nUnified latency report saved to: {args.output}")

    csv_path = args.output.replace('.json', '.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'dataset', 'setting', 'index_type', 'k', 'pool_size',
                    'n_queries', 'runs_per_query',
                    't_text_encode_mean_ms', 't_text_encode_std_ms',
                    't_search_mean_ms', 't_search_std_ms',
                    't_rerank_mean_ms', 't_rerank_std_ms',
                    't_total_mean_ms'])
        for s in summaries:
            w.writerow([
                s['label'], s['dataset'], s['setting'], s['index_type'],
                s['k'], s['pool_size'], s['num_queries_timed'],
                s['num_runs_per_query'],
                s['t_text_encode_ms']['mean'], s['t_text_encode_ms']['std'],
                s['t_search_ms']['mean'], s['t_search_ms']['std'],
                s['t_rerank_ms']['mean'], s['t_rerank_ms']['std'],
                s['t_total_ms']['mean'],
            ])
    print(f"CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
