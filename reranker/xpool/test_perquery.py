import os
import sys
import csv
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from trainer.evaluator import PerQueryEvaluator
from utils.checkpoint import load_state_dict_compat
import argparse


def summarize_timing_stats(timing_stats):
    """Return mean/std/min/max timing stats in seconds for each timing key."""
    summary = {}
    for key, values in timing_stats.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            'mean_s': float(arr.mean()),
            'std_s': float(arr.std()),
            'min_s': float(arr.min()),
            'max_s': float(arr.max()),
        }
    return summary


def build_summary_row(config, custom_args, evaluator, metrics, timing_summary, candidates_file):
    """Build one structured summary row with metrics and timing breakdowns."""
    candidate_counts = [len(v) for v in evaluator.query_candidates.values()]
    row = {
        'dataset': config.dataset_name.lower(),
        'retrieval_mode': 'candidates' if candidates_file else (
            'expanded_pool' if custom_args.expanded_pool else 'full_test_pool'
        ),
        'candidate_file': candidates_file or '',
        'cache_dir': custom_args.cache_dir or '',
        'num_frames': config.num_frames,
        'num_queries': evaluator.query_count,
        'search_pool_size': len(evaluator.video_ids),
        'candidate_count_mean': float(np.mean(candidate_counts)) if candidate_counts else '',
        'candidate_count_min': int(np.min(candidate_counts)) if candidate_counts else '',
        'candidate_count_max': int(np.max(candidate_counts)) if candidate_counts else '',
        'R@1': float(metrics['R1']),
        'R@5': float(metrics['R5']),
        'R@10': float(metrics['R10']),
        'R@50': float(metrics['R50']),
        'R@100': float(metrics['R100']),
        'MedR': float(metrics['MedR']),
        'MeanR': float(metrics['MeanR']),
    }
    for key in ('query_encode', 'video_load', 'frame_pooling', 'similarity_compute', 'total'):
        stats = timing_summary.get(key)
        row[f'{key}_ms_mean'] = float(stats['mean_s'] * 1000.0) if stats else ''
        row[f'{key}_ms_std'] = float(stats['std_s'] * 1000.0) if stats else ''
        row[f'{key}_ms_min'] = float(stats['min_s'] * 1000.0) if stats else ''
        row[f'{key}_ms_max'] = float(stats['max_s'] * 1000.0) if stats else ''
    return row


def write_summary_csv(path, row):
    """Write one-row CSV summary."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def load_test_queries(config):
    """
    Load test queries and ground truth video IDs from the test dataset.

    Returns:
        List of (query_text, video_id) tuples
    """
    # Load test data based on dataset
    if config.dataset_name == "MSRVTT":
        test_csv = 'reranker/xpool/data/MSRVTT/MSRVTT_JSFUSION_test.csv'
        test_df = pd.read_csv(test_csv)
        queries = [(row.sentence, row.video_id) for _, row in test_df.iterrows()]

    elif config.dataset_name == "MSVD":
        test_csv = 'reranker/xpool/data/MSVD/MSVD_test.csv'
        test_df = pd.read_csv(test_csv)
        queries = [(row.sentence, row.video_id) for _, row in test_df.iterrows()]

    elif config.dataset_name == "LSMDC":
        test_csv = 'reranker/xpool/data/LSMDC/LSMDC16_challenge_1000_publictect.csv'
        queries = []
        with open(test_csv, 'r') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split("\t")
                assert len(line_split) == 6
                clip_id, _, _, _, _, caption = line_split
                # Skip the problematic clip (consistent with lsmdc_dataset.py)
                if clip_id == '1012_Unbreakable_00.05.16.065-00.05.21.941':
                    continue
                queries.append((caption, clip_id))

    elif config.dataset_name == "ACTNET":
        test_json = 'reranker/xpool/data/ACTNET/actnet_ret_test.json'
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        queries = []
        for item in test_data:
            video_id = item['video'].replace('.mp4', '')
            caption = ' '.join(c.strip() for c in item['caption'])
            queries.append((caption, video_id))

    elif config.dataset_name == "DIDEMO":
        test_json = 'reranker/xpool/data/DIDEMO/didemo_ret_test.json'
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        queries = []
        for item in test_data:
            video_id = item['video'].replace('.mp4', '')
            caption = ' '.join(c.strip() for c in item['caption'])
            queries.append((caption, video_id))

    else:
        raise NotImplementedError(f"Dataset {config.dataset_name} not supported")

    return queries


def load_queries_from_candidates_file(candidates_file):
    """Load query text and GT video IDs directly from candidate JSON."""
    with open(candidates_file, 'r') as f:
        payload = json.load(f)

    if 'results' not in payload:
        raise ValueError(f"Candidates file missing 'results' key: {candidates_file}")

    return [
        (item['query_text'], item['ground_truth_video_id'])
        for item in payload['results']
    ]


def get_unique_video_ids(queries):
    """
    Extract unique video IDs from query list while preserving order.

    Args:
        queries: List of (query_text, video_id) tuples

    Returns:
        List of unique video IDs in order of first appearance
    """
    seen = set()
    unique_ids = []

    for _, video_id in queries:
        if video_id not in seen:
            seen.add(video_id)
            unique_ids.append(video_id)

    return unique_ids


def main():
    # Load configuration first (AllConfig has its own arg parser)
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # Parse additional custom arguments not in AllConfig
    custom_parser = argparse.ArgumentParser(description='Per-query evaluation', add_help=False)
    custom_parser.add_argument('--cache_dir', type=str, default="reranker/xpool/video_features_cache/Xpool/ACTNET",
                        help='Directory with cached video features (None for on-the-fly mode)')
    custom_parser.add_argument('--max_queries', type=int,
                        help='Maximum number of queries to evaluate (for testing)')
    custom_parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save detailed per-query results (JSON)')
    custom_parser.add_argument('--checkpoint', type=str,
                        default='reranker/xpool/ckpt/actnet_model_best.pth',
                        help='Path to model checkpoint')
    custom_parser.add_argument('--expanded_pool', action='store_true',
                        help='Add training videos to search pool')
    custom_parser.add_argument('--report_dir', type=str, default='output/evaluation_results/rerank',
                        help='Directory for CSV/JSON reports')
    custom_parser.add_argument('--summary_csv', type=str, default=None,
                        help='Optional path for structured summary CSV')
    custom_parser.add_argument('--summary_json', type=str, default=None,
                        help='Optional path for structured summary JSON')
    custom_parser.add_argument('--index_safe_candidates', action='store_true',
                        help='Use candidate file row index instead of query text when selecting per-query candidates')
    # Pass-B (efficiency) latency contract --- see
    # research_html/packages/2026-05-15-panda-baselines/docs/eval-efficiency.html
    custom_parser.add_argument('--subset_manifest', type=str, default=None,
                        help='Pass-B latency manifest with warmup_query_ids + timed_query_ids')
    custom_parser.add_argument('--warmup_n_used', type=int, default=10,
                        help='How many warmup ids from the manifest to consume (EERCF wrapper overrides to 1)')
    custom_parser.add_argument('--wall_time_cap_s', type=float, default=300.0,
                        help='Per-cell wall-time cap; stops between queries when elapsed exceeds this')
    custom_parser.add_argument('--latency_helpers_dir', type=str,
                        default='research_html/packages/2026-05-15-panda-baselines/scripts',
                        help='Path containing latency_helpers.py')
    custom_parser.add_argument('--skip_cache_miss', action='store_true',
                        help='Drop candidates that miss the .npz cache instead of '
                             'falling back to on-the-fly extraction. Required when '
                             'rerank consumes candidate JSONs from a baseline whose '
                             'id format differs from the X-Pool cache layout (e.g. '
                             'EERCF-LSMDC doubled-prefix ids).')

    custom_args, _ = custom_parser.parse_known_args()
    if custom_args.cache_dir in ("", "none", "None", "null", "NULL"):
        custom_args.cache_dir = None

    # Get candidates_file from AllConfig (uses --candidate_file argument)
    # candidates_file = "candidates/msrvtt_videorqvae__c128l3_100_candidates.json"
    # candidates_file = None
    candidates_file = config.candidate_file

    # Set random seed for reproducibility
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("="*70)
    print("Per-Query Evaluation")
    print("="*70)
    print(f"Dataset: {config.dataset_name}")
    print(f"Model checkpoint: {custom_args.checkpoint}")
    if candidates_file:
        print(f"Retrieval mode: Candidate Reranking")
        print(f"Candidates file: {candidates_file}")
    elif custom_args.expanded_pool:
        print(f"Retrieval mode: Full Retrieval (expanded pool: test + train)")
    else:
        print(f"Retrieval mode: Full Retrieval (test only)")
    print(f"Feature mode: {'Cached' if custom_args.cache_dir else 'On-the-fly'}")
    if custom_args.cache_dir:
        print(f"Cache directory: {custom_args.cache_dir}")
    print(f"Videos directory: {config.videos_dir}")
    print("="*70)

    # Load tokenizer
    if config.huggingface:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",
                                                   TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    # Load model
    print("\nLoading model...")
    model = ModelFactory.get_model(config)

    # Load checkpoint
    if os.path.exists(custom_args.checkpoint):
        print(f"Loading checkpoint: {custom_args.checkpoint}")
        load_state_dict_compat(model, custom_args.checkpoint)
    else:
        print(f"Warning: Checkpoint not found: {custom_args.checkpoint}")

    # Load queries
    if candidates_file:
        print("\nLoading queries from candidate file...")
        queries = load_queries_from_candidates_file(candidates_file)
    else:
        print("\nLoading test queries...")
        queries = load_test_queries(config)
    print(f"Total queries: {len(queries)}")

    # Get unique video IDs (only needed for full retrieval mode)
    # IMPORTANT: Extract video IDs BEFORE limiting queries to ensure fair evaluation
    if candidates_file is None:
        unique_video_ids = get_unique_video_ids(queries)
        print(f"Unique videos in test set: {len(unique_video_ids)}")
        
        # Expand pool with train videos if flag is set
        if custom_args.expanded_pool:
            from datasets.data_factory import DataFactory
            train_vid_ids, _, _, _ = DataFactory.get_train_video_ids(config)
            num_test_vids = len(unique_video_ids)
            unique_video_ids = unique_video_ids + train_vid_ids
            print(f"Expanded pool enabled: {num_test_vids} test + {len(train_vid_ids)} train = {len(unique_video_ids)} total videos")
    else:
        # In candidate mode, video_ids will be loaded from candidates file
        unique_video_ids = []

    # Limit queries if specified (for testing/debugging)
    # This should be done AFTER extracting video IDs to maintain fair search pool
    if custom_args.max_queries is not None:
        queries = queries[:custom_args.max_queries]
        print(f"Evaluating first {len(queries)} queries (search pool unchanged)")

    # Pass-B latency mode: load subset manifest and split queries into
    # (warmup, timed) preserving the manifest's qid ordering.
    latency_manifest = None
    warmup_queries: list = []
    if custom_args.subset_manifest:
        sys.path.insert(0, custom_args.latency_helpers_dir)
        from latency_helpers import load_subset_manifest  # noqa: E402
        latency_manifest = load_subset_manifest(custom_args.subset_manifest)

        wanted_warmup = list(latency_manifest['warmup_query_ids'][: custom_args.warmup_n_used])
        wanted_timed = list(latency_manifest['timed_query_ids'])
        # qid = video_id; duplicate video_ids in the loader are suffixed #dupN.
        by_qid: dict = {}
        dup = {}
        for t, v in queries:
            dup[v] = dup.get(v, 0) + 1
            key = v if dup[v] == 1 else f"{v}#dup{dup[v]}"
            by_qid[key] = (t, v)
        warmup_queries = [by_qid[q] for q in wanted_warmup if q in by_qid]
        queries = [by_qid[q] for q in wanted_timed if q in by_qid]
        missing_w = [q for q in wanted_warmup if q not in by_qid]
        missing_t = [q for q in wanted_timed if q not in by_qid]
        if missing_w or missing_t:
            print(f"Pass-B WARN: manifest qids missing from X-Pool loader — "
                  f"warmup={len(missing_w)} timed={len(missing_t)}")
        print(f"Pass-B subset: warmup={len(warmup_queries)} timed={len(queries)}"
              f" (manifest sha={latency_manifest['metadata'].get('content_sha256','')[:10]})")

    # Define excluded videos per dataset (problematic videos to skip)
    excluded_videos = []
    if config.dataset_name == "LSMDC":
        excluded_videos = ['1012_Unbreakable_00.05.16.065-00.05.21.941']
    
    # Initialize evaluator
    print("\nInitializing PerQueryEvaluator...")
    evaluator = PerQueryEvaluator(
        model=model,
        config=config,
        video_ids=unique_video_ids,
        tokenizer=tokenizer,
        cache_dir=custom_args.cache_dir,
        videos_dir=config.videos_dir,
        candidates_file=candidates_file,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        excluded_videos=excluded_videos,
        skip_cache_miss=custom_args.skip_cache_miss,
    )

    # Evaluate queries one at a time
    print("\nEvaluating queries...")
    print("-"*70)

    per_query_results = []

    # Pass-B latency mode: run warmup queries first (timings discarded by
    # resetting the evaluator's timing_stats), then time the subset with a
    # wall-time cap and finally restore the per-query timings.
    if latency_manifest is not None:
        import time as _time
        print(f"Warmup ({len(warmup_queries)} queries) ...")
        for query_text, video_id_gt in warmup_queries:
            evaluator.evaluate_query(query_text, video_id_gt, candidate_query_idx=None)
        for key in evaluator.timing_stats:
            evaluator.timing_stats[key] = []
        evaluator.query_count = 0
        evaluator.query_texts = []
        evaluator.query_video_ids = []
        evaluator.query_candidates = {}
        evaluator.query_similarities = {}
        per_query_results = []
        cap_t0 = _time.perf_counter()
        cap_hit = False
        for candidate_query_idx, (query_text, video_id_gt) in enumerate(
            tqdm(queries, desc="Pass-B timed queries")
        ):
            if (_time.perf_counter() - cap_t0) >= custom_args.wall_time_cap_s:
                cap_hit = True
                break
            result = evaluator.evaluate_query(
                query_text,
                video_id_gt,
                candidate_query_idx=candidate_query_idx if (
                    custom_args.index_safe_candidates and candidates_file
                ) else None,
            )
            per_query_results.append({
                'query_idx': result['query_idx'],
                'candidate_query_idx': result.get('candidate_query_idx'),
                'query_text': result['query_text'],
                'video_id_gt': result['video_id_gt'],
                'rank': result['rank'],
                'top_5_videos': result['ranked_videos'][:5],
                'candidate_count': result.get('candidate_count'),
                'timing': result['timing'],
            })
        latency_wall_seconds = _time.perf_counter() - cap_t0
        latency_n_processed = len(per_query_results)
        latency_n_target = len(queries)
        latency_cap_hit = cap_hit
    else:
        for candidate_query_idx, (query_text, video_id_gt) in enumerate(tqdm(queries, desc="Processing queries")):
            # Evaluate single query
            result = evaluator.evaluate_query(
                query_text,
                video_id_gt,
                candidate_query_idx=candidate_query_idx if (
                    custom_args.index_safe_candidates and candidates_file
                ) else None,
            )

            # Store detailed results
            per_query_results.append({
                'query_idx': result['query_idx'],
                'candidate_query_idx': result.get('candidate_query_idx'),
                'query_text': result['query_text'],
                'video_id_gt': result['video_id_gt'],
                'rank': result['rank'],
                'top_5_videos': result['ranked_videos'][:5],
                'candidate_count': result.get('candidate_count'),
                'timing': result['timing']
            })

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)

    # Compute final metrics
    print("\nComputing final metrics...")
    metrics = evaluator.compute_final_metrics()
    timing_summary = summarize_timing_stats(evaluator.timing_stats)

    # Display results
    print("\n" + "="*70)
    print("RETRIEVAL METRICS")
    print("="*70)
    print(f"R@1:   {metrics['R1']:.2f}%")
    print(f"R@5:   {metrics['R5']:.2f}%")
    print(f"R@10:  {metrics['R10']:.2f}%")
    print(f"R@50:  {metrics['R50']:.2f}%")
    print(f"R@100: {metrics['R100']:.2f}%")
    print(f"MedR:  {metrics['MedR']:.2f}")
    print(f"MeanR: {metrics['MeanR']:.2f}")
    print("="*70)

    # Display timing statistics
    print("\n" + evaluator.get_timing_summary())
    print("="*70)

    # Save results to CSV
    report_dir = custom_args.report_dir
    os.makedirs(report_dir, exist_ok=True)

    csv_path = os.path.join(report_dir, f"perquery_{config.dataset_name.lower()}_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['R@1', 'R@5', 'R@10', 'R@50', 'R@100', 'MedR', 'MeanR'])
        # Write values
        csv_writer.writerow([
            metrics['R1'],
            metrics['R5'],
            metrics['R10'],
            metrics['R50'],
            metrics['R100'],
            metrics['MedR'],
            metrics['MeanR']
        ])

    print(f"\nResults saved to: {csv_path}")

    summary_row = build_summary_row(
        config=config,
        custom_args=custom_args,
        evaluator=evaluator,
        metrics=metrics,
        timing_summary=timing_summary,
        candidates_file=candidates_file,
    )
    summary_csv_path = custom_args.summary_csv or os.path.join(
        report_dir, f"perquery_{config.dataset_name.lower()}_summary.csv"
    )
    write_summary_csv(summary_csv_path, summary_row)
    print(f"Structured summary saved to: {summary_csv_path}")

    summary_json_path = custom_args.summary_json or os.path.join(
        report_dir, f"perquery_{config.dataset_name.lower()}_summary.json"
    )
    summary_payload = {
        'summary': summary_row,
        'timing_summary': timing_summary,
        'metrics': {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in metrics.items() if k != 'timing_avg'
        },
        'config': {
            'dataset': config.dataset_name,
            'retrieval_mode': summary_row['retrieval_mode'],
            'candidate_file': candidates_file,
            'index_safe_candidates': custom_args.index_safe_candidates,
            'num_frames': config.num_frames,
            'cache_dir': custom_args.cache_dir,
            'videos_dir': config.videos_dir,
            'search_pool_size': len(evaluator.video_ids),
        }
    }
    if latency_manifest is not None:
        sys.path.insert(0, custom_args.latency_helpers_dir)
        from latency_helpers import host_fingerprint  # noqa: E402
        total_summary = timing_summary.get('total', {})
        component_means = {
            k: float(timing_summary.get(k, {}).get('mean_s', 0.0) * 1000.0)
            for k in ('query_encode', 'video_load', 'frame_pooling', 'similarity_compute')
        }
        summary_payload['rerank_latency_ms'] = {
            'online_total_mean': float(total_summary.get('mean_s', 0.0) * 1000.0),
            'online_total_p95': float(np.percentile(
                np.asarray(evaluator.timing_stats['total']) * 1000.0, 95
            )) if evaluator.timing_stats['total'] else 0.0,
            'online_total_std': float(total_summary.get('std_s', 0.0) * 1000.0),
            'component_breakdown_mean_ms': component_means,
            'n_processed': int(latency_n_processed),
            'n_target': int(latency_n_target),
            'warmup_n': len(latency_manifest.get('warmup_query_ids', [])),
            'warmup_n_used': int(custom_args.warmup_n_used),
            'validity': 'full_subset' if (
                latency_n_processed >= latency_n_target and not latency_cap_hit
            ) else 'truncated_subset',
            'wall_seconds': float(latency_wall_seconds),
            'cap_hit': bool(latency_cap_hit),
            'strict_latency_contract': 'xpool_total_per_query_cuda_sync',
            'latency_batch_size': 1,
        }
        # Pass-B provenance fields at metadata top level (spec).
        summary_payload['method'] = 'xpool_rerank'
        summary_payload['setting'] = 2 if getattr(custom_args, 'expanded_pool', False) else 1
        summary_payload['subset_manifest'] = custom_args.subset_manifest
        summary_payload['subset_manifest_sha256'] = latency_manifest['metadata'].get('content_sha256', '')
        summary_payload['host_fingerprint'] = host_fingerprint()
    with open(summary_json_path, 'w') as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Structured JSON saved to: {summary_json_path}")

    # Save detailed per-query results if requested
    if custom_args.save_results:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable_results = []
        for result in per_query_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    serializable_result[key] = {k: convert_to_serializable(v)
                                                for k, v in value.items()}
                elif isinstance(value, list):
                    serializable_result[key] = [
                        (convert_to_serializable(item[0]), convert_to_serializable(item[1]))
                        if isinstance(item, tuple) else convert_to_serializable(item)
                        for item in value
                    ]
                else:
                    serializable_result[key] = convert_to_serializable(value)
            serializable_results.append(serializable_result)

        with open(custom_args.save_results, 'w') as f:
            json.dump({
                'metrics': {k: convert_to_serializable(v) for k, v in metrics.items()
                           if k != 'timing_avg'},
                'per_query_results': serializable_results,
                'config': {
                    'dataset': config.dataset_name,
                    'pooling_type': config.pooling_type,
                    'num_frames': config.num_frames,
                    'index_safe_candidates': custom_args.index_safe_candidates,
                    'feature_mode': 'cached' if custom_args.cache_dir else 'on-the-fly'
                }
            }, f, indent=2)

        print(f"Detailed results saved to: {custom_args.save_results}")

    print("\nDone!")


if __name__ == '__main__':
    main()
