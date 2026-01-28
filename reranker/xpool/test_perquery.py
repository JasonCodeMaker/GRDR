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
import argparse


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

    custom_args, _ = custom_parser.parse_known_args()

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
        checkpoint = torch.load(custom_args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"Warning: Checkpoint not found: {custom_args.checkpoint}")

    # Load test queries
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
        excluded_videos=excluded_videos
    )

    # Evaluate queries one at a time
    print("\nEvaluating queries...")
    print("-"*70)

    per_query_results = []

    for query_text, video_id_gt in tqdm(queries, desc="Processing queries"):
        # Evaluate single query
        result = evaluator.evaluate_query(query_text, video_id_gt)

        # Store detailed results
        per_query_results.append({
            'query_idx': result['query_idx'],
            'query_text': result['query_text'],
            'video_id_gt': result['video_id_gt'],
            'rank': result['rank'],
            'top_5_videos': result['ranked_videos'][:5],
            'timing': result['timing']
        })

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)

    # Compute final metrics
    print("\nComputing final metrics...")
    metrics = evaluator.compute_final_metrics()

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
    output_dir = "output/reranker"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"perquery_{config.dataset_name.lower()}_results.csv")
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
                    'feature_mode': 'cached' if custom_args.cache_dir else 'on-the-fly'
                }
            }, f, indent=2)

        print(f"Detailed results saved to: {custom_args.save_results}")

    print("\nDone!")


if __name__ == '__main__':
    main()
