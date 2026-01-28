#!/usr/bin/env python3
"""
Verify and regenerate pseudo labels for failed videos.

This script:
1. Identifies videos with 0 pseudo labels (failed generation)
2. Regenerates pseudo labels using the same batch API pipeline
3. Merges new results back into the pseudo labels file
"""

import os
import sys
import json
import time
import argparse
from collections import Counter, defaultdict

# Import reusable functions from the original generation script
sys.path.insert(0, os.path.dirname(__file__))
from gen_pesudo_labe_batch import (
    sample_video_frames,
    encode_frames_base64,
    build_gpt_prompt,
    create_batch_request,
    submit_batch,
    poll_batch_completion,
    parse_batch_results,
    save_pseudo_labels
)

from openai import OpenAI
from tqdm import tqdm


def load_train_data(dataset_path):
    """Load training data and group captions by video ID."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    video_captions = defaultdict(list)
    for item in data:
        video_id = item['video']
        caption = item['caption']
        video_captions[video_id].append(caption)

    return dict(video_captions)


def verify_pseudo_labels(original_data_path, pseudo_data_path, target_total_count=50):
    """
    Verify pseudo labels and identify failed videos based on total captions.
    
    Args:
        original_data_path: Path to original training data
        pseudo_data_path: Path to pseudo labels
        target_total_count: Expected total number of captions per video (original + pseudo)
        
    Returns:
        Dictionary with verification results
    """
    print(f"{'='*60}")
    print(f"Verification Report")
    print(f"{'='*60}")
    
    # Load original data
    with open(original_data_path, 'r') as f:
        train_data = json.load(f)
    
    # Load pseudo labels
    if os.path.exists(pseudo_data_path):
        with open(pseudo_data_path, 'r') as f:
            pseudo_data = json.load(f)
    else:
        pseudo_data = []
        print(f"Warning: Pseudo labels file not found at {pseudo_data_path}")
    
    # Count captions per video
    train_counts = Counter(item['video'] for item in train_data)
    pseudo_counts = Counter(item['video'] for item in pseudo_data)
    
    print(f"\nOriginal training data:")
    print(f"  Total entries: {len(train_data)}")
    print(f"  Unique videos: {len(train_counts)}")
    print(f"  Avg captions per video: {len(train_data) / len(train_counts):.1f}")
    
    print(f"\nPseudo labels:")
    print(f"  Total entries: {len(pseudo_data)}")
    print(f"  Unique videos: {len(pseudo_counts)}")
    if len(pseudo_counts) > 0:
        print(f"  Avg captions per video: {len(pseudo_data) / len(pseudo_counts):.1f}")
    
    # Categorize videos
    missing_videos = []
    incomplete_videos = []
    complete_videos = []
    overcomplete_videos = []
    
    for video_id in train_counts:
        original_cnt = train_counts[video_id]
        pseudo_cnt = pseudo_counts.get(video_id, 0)
        total_cnt = original_cnt + pseudo_cnt
        
        if pseudo_cnt == 0:
            missing_videos.append(video_id)
        elif total_cnt < target_total_count:
            incomplete_videos.append((video_id, total_cnt))
        elif total_cnt == target_total_count:
            complete_videos.append(video_id)
        else:
            overcomplete_videos.append((video_id, total_cnt))
    
    print(f"\nTarget total captions per video: {target_total_count}")
    print(f"\nStatus breakdown:")
    print(f"  Missing (0 pseudo captions): {len(missing_videos)} videos")
    print(f"  Incomplete (< {target_total_count} total captions): {len(incomplete_videos)} videos")
    print(f"  Complete (= {target_total_count} total captions): {len(complete_videos)} videos")
    print(f"  Overcomplete (> {target_total_count} total captions): {len(overcomplete_videos)} videos")
    
    if missing_videos:
        print(f"\nSample missing videos (first 10):")
        for vid in missing_videos[:10]:
            print(f"  {vid}")
    
    if incomplete_videos:
        print(f"\nSample incomplete videos (first 10):")
        for vid, cnt in incomplete_videos[:10]:
            print(f"  {vid}: {cnt} total captions (need {target_total_count - cnt} more)")
    
    if overcomplete_videos:
        print(f"\nOvercomplete videos:")
        for vid, cnt in overcomplete_videos:
            print(f"  {vid}: {cnt} total captions")
    
    print(f"\n{'='*60}")
    
    return {
        'missing_videos': missing_videos,
        'incomplete_videos': incomplete_videos,
        'complete_videos': complete_videos,
        'overcomplete_videos': overcomplete_videos,
        'train_counts': train_counts,
        'pseudo_counts': pseudo_counts
    }


def prepare_regeneration_requests(video_ids, video_captions, args):
    """
    Prepare batch requests for videos needing regeneration.
    
    Args:
        video_ids: List of video IDs to regenerate
        video_captions: Dict mapping video_id to list of existing captions
        args: Command line arguments
        
    Returns:
        List of (video_id, request_dict, num_to_generate) tuples
    """
    requests = []
    
    for video_id in tqdm(video_ids, desc="Preparing regeneration requests"):
        try:
            existing_captions = video_captions.get(video_id, [])
            num_to_generate = max(0, args.target_captions - len(existing_captions))
            
            if num_to_generate == 0:
                print(f"Skipping {video_id}: already has {len(existing_captions)} captions")
                continue
            
            # Find video file
            video_path = os.path.join(args.video_dir, video_id)
            if not os.path.exists(video_path):
                base_name = os.path.splitext(video_id)[0]
                for ext in ['mp4', 'avi', 'mov', 'mkv']:
                    alt_path = os.path.join(args.video_dir, f"{base_name}.{ext}")
                    if os.path.exists(alt_path):
                        video_path = alt_path
                        break
            
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_id}")
                continue
            
            # Sample and encode frames
            frames = sample_video_frames(video_path, args.num_frames)
            if frames is None:
                print(f"Warning: Failed to extract frames from {video_id}")
                continue
            
            frames_b64 = encode_frames_base64(frames)
            prompt = build_gpt_prompt(existing_captions if existing_captions else ["video content"], num_to_generate)
            
            # Create batch request
            request = create_batch_request(video_id, frames_b64, prompt, args.model)
            requests.append((video_id, request, num_to_generate))
            
        except Exception as e:
            print(f"Error preparing video {video_id}: {e}")
            continue
    
    return requests


def create_batch_file(requests, batch_file_path):
    """Create JSONL file for batch API submission."""
    with open(batch_file_path, 'w') as f:
        for video_id, request, _ in requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created batch file with {len(requests)} requests: {batch_file_path}")


def merge_pseudo_labels(existing_path, new_labels):
    """
    Merge new pseudo labels with existing ones.
    
    Args:
        existing_path: Path to existing pseudo labels JSON
        new_labels: List of new pseudo label dicts to add
        
    Returns:
        Combined list of pseudo labels
    """
    # Load existing labels
    if os.path.exists(existing_path):
        with open(existing_path, 'r') as f:
            existing_labels = json.load(f)
    else:
        existing_labels = []
    
    print(f"Existing pseudo labels: {len(existing_labels)}")
    print(f"New pseudo labels: {len(new_labels)}")
    
    # Merge
    combined = existing_labels + new_labels
    
    print(f"Combined pseudo labels: {len(combined)}")
    
    return combined


def parse_args():
    parser = argparse.ArgumentParser(description='Verify and regenerate pseudo labels for failed videos')
    parser.add_argument('--dataset', type=str, default='didemo', choices=['msrvtt', 'didemo', 'actnet', 'lsmdc'],
                        help='Dataset to process')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root data directory')
    parser.add_argument('--video_dir', type=str, default='data_process/datasets/DiDeMo/train/videos',
                        help='Directory containing video files')
    parser.add_argument('--target_captions', type=int, default=50,
                        help='Target total captions per video (original + pseudo)')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames to sample per video')
    parser.add_argument('--poll_interval', type=int, default=60,
                        help='Batch status polling interval in seconds')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Maximum requests per batch')
    parser.add_argument('--verify_only', action='store_true', default=False,
                        help='Only run verification, do not regenerate')
    parser.add_argument('--mode', type=str, default='strict', choices=['strict', 'all_incomplete'],
                        help='Regeneration mode: strict (only 0 pseudo) or all_incomplete (< target)')
    parser.add_argument('--model', type=str, default='gpt-5-mini-2025-08-07', # gpt-5-mini-2025-08-07, gpt-5-2025-08-07
                        help='OpenAI model to use')
    return parser.parse_args()


def main():
    """Main function for verification and regeneration."""
    args = parse_args()
    
    # Setup paths
    data_path = os.path.join(args.data_dir, args.dataset, 'video_retreival_caption', f'{args.dataset}_ret_train.json')
    pseudo_path = os.path.join(args.data_dir, args.dataset, 'video_retreival_caption', f'{args.dataset}_ret_train_pesudo.json')
    batch_dir = os.path.join(args.data_dir, args.dataset, 'video_retreival_caption', 'batch_files_regen')
    os.makedirs(batch_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Pseudo Label Verification and Regeneration")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Original data: {data_path}")
    print(f"Pseudo labels: {pseudo_path}")
    print(f"Video directory: {args.video_dir}")
    
    # Step 1: Verify existing pseudo labels
    verification_results = verify_pseudo_labels(data_path, pseudo_path, args.target_captions)
    
    # Determine which videos to regenerate based on mode
    if args.mode == 'strict':
        videos_to_regenerate = verification_results['missing_videos']
        print(f"\nMode: strict - Regenerating {len(videos_to_regenerate)} videos with 0 pseudo labels")
    elif args.mode == 'all_incomplete':
        videos_to_regenerate = verification_results['missing_videos']
        incomplete = [vid for vid, _ in verification_results['incomplete_videos']]
        videos_to_regenerate.extend(incomplete)
        print(f"\nMode: all_incomplete - Regenerating {len(videos_to_regenerate)} videos")
        print(f"  {len(verification_results['missing_videos'])} missing + {len(incomplete)} incomplete")
    
    if len(videos_to_regenerate) == 0:
        print("\nNo videos need regeneration. All done!")
        return
    
    if args.verify_only:
        print("\nVerification complete. Use --mode strict or --mode all_incomplete without --verify_only to regenerate.")
        return
    
    # Verify API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Load original captions for prompt context
    video_captions = load_train_data(data_path)
    
    print(f"\n{'='*60}")
    print(f"Step 2: Preparing Regeneration Requests")
    print(f"{'='*60}")
    
    # Prepare requests
    requests_info = prepare_regeneration_requests(videos_to_regenerate, video_captions, args)
    
    if len(requests_info) == 0:
        print("No valid videos to regenerate (all video files missing or unreadable)")
        return
    
    print(f"Successfully prepared {len(requests_info)} requests")
    
    # Split into batches
    num_batches = (len(requests_info) + args.batch_size - 1) // args.batch_size
    print(f"Will submit {num_batches} batch(es) to OpenAI")
    
    # Process each batch
    all_new_pseudo_labels = []
    
    for batch_idx in range(num_batches):
        batch_start_idx = batch_idx * args.batch_size
        batch_end_idx = min((batch_idx + 1) * args.batch_size, len(requests_info))
        batch_requests = requests_info[batch_start_idx:batch_end_idx]
        
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_idx + 1}/{num_batches}")
        print(f"Requests: {len(batch_requests)}")
        print(f"{'='*60}")
        
        # Create batch file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_file_path = os.path.join(batch_dir, f'regen_batch_{timestamp}_{batch_idx}.jsonl')
        create_batch_file(batch_requests, batch_file_path)
        
        # Submit batch
        print(f"\nSubmitting batch to OpenAI...")
        batch_description = f"{args.dataset}_regen_batch_{batch_idx}"
        batch = submit_batch(client, batch_file_path, batch_description)
        
        # Poll for completion
        print(f"\nWaiting for batch completion...")
        completed_batch = poll_batch_completion(client, batch.id, args.poll_interval)
        
        if completed_batch.status != "completed":
            print(f"Warning: Batch {batch_idx + 1} did not complete successfully!")
            print(f"Status: {completed_batch.status}")
            continue
        
        # Parse results
        print(f"\nParsing batch results...")
        new_labels = parse_batch_results(client, completed_batch, batch_requests)
        all_new_pseudo_labels.extend(new_labels)
        
        print(f"Batch {batch_idx + 1} generated {len(new_labels)} pseudo labels")
    
    print(f"\n{'='*60}")
    print(f"Step 3: Merging Results")
    print(f"{'='*60}")
    
    # Merge with existing pseudo labels
    combined_labels = merge_pseudo_labels(pseudo_path, all_new_pseudo_labels)
    
    # Save merged results
    backup_path = pseudo_path + f".backup_{time.strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(pseudo_path):
        print(f"Creating backup: {backup_path}")
        os.system(f"cp {pseudo_path} {backup_path}")
    
    save_pseudo_labels(combined_labels, pseudo_path)
    
    # Final verification
    print(f"\n{'='*60}")
    print(f"Final Verification")
    print(f"{'='*60}")
    verify_pseudo_labels(data_path, pseudo_path, args.target_captions)
    
    print(f"\n{'='*60}")
    print(f"Regeneration Complete!")
    print(f"{'='*60}")
    print(f"Videos regenerated: {len(videos_to_regenerate)}")
    print(f"New pseudo labels generated: {len(all_new_pseudo_labels)}")
    print(f"Total pseudo labels: {len(combined_labels)}")
    print(f"Output saved to: {pseudo_path}")
    print(f"Backup saved to: {backup_path}")


if __name__ == "__main__":
    main()

