import os
import json
import base64
import time
import argparse
from io import BytesIO
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader
from openai import OpenAI


def load_train_data(dataset_path):
    """Load training data and group captions by video ID.
    
    Handles both formats:
    - MSRVTT/LSMDC: {"video": "...", "caption": "string"} (one entry per caption)
    - DiDeMo/ActivityNet: {"video": "...", "caption": ["list", "of", "captions"]}
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    video_captions = defaultdict(list)
    for item in data:
        video_id = item['video']
        caption = item['caption']
        # Handle both single caption string and list of captions
        if isinstance(caption, list):
            video_captions[video_id].extend(caption)
        else:
            video_captions[video_id].append(caption)

    return dict(video_captions)


def sample_video_frames(video_path, num_frames=8):
    """Sample frames uniformly from video using decord."""
    try:
        vr = VideoReader(video_path, num_threads=1)
        vlen = len(vr)

        if vlen <= num_frames:
            frame_indices = np.arange(vlen)
        else:
            frame_indices = np.linspace(0, vlen - 1, num_frames, dtype=int)

        frames = vr.get_batch(frame_indices).asnumpy()
        return [Image.fromarray(f) for f in frames]

    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None


def encode_frames_base64(frames, quality=85, max_size=512):
    """Encode PIL frames to base64 JPEG strings with resizing for API efficiency."""
    encoded = []
    for frame in frames:
        # Resize to reduce API costs while maintaining quality
        w, h = frame.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            frame = frame.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buffer = BytesIO()
        frame.save(buffer, format="JPEG", quality=quality)
        encoded.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    return encoded


def build_gpt_prompt(existing_captions, num_to_generate):
    """Build the prompt for GPT to generate diverse captions."""
    captions_text = "\n".join(f"- {c}" for c in existing_captions[:10])  # Limit to 10 examples

    prompt = f"""Given these 8 frames from a video and the existing captions below, generate EXACTLY {num_to_generate} diverse NEW captions that:
1. Describe the video content accurately based on what you see
2. Mantain similar sentence structures as existing captions but use different vocabulary and combinations of words
3. Focus on different aspects (actions, objects, settings, details)
4. Maintain similar length and style to existing captions (1 sentence, lowercase start)

Existing captions:
{captions_text}

IMPORTANT: Return ONLY a JSON array containing EXACTLY {num_to_generate} new caption strings. No more, no less. No explanations, no markdown formatting, just the JSON array with exactly {num_to_generate} elements."""

    return prompt


def create_batch_request(custom_id, frames_b64, prompt, model="gpt-4o-mini"):
    """Create a single batch request for OpenAI Batch API.
    
    Args:
        custom_id: Unique identifier for this request (video_id)
        frames_b64: Base64 encoded frames
        prompt: Prompt for the API
        model: OpenAI model to use (default: gpt-4o-mini)
        
    Returns:
        Dictionary representing a single batch request
    """
    image_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}", "detail": "low"}}
        for f in frames_b64
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates diverse, accurate video descriptions."},
        {"role": "user", "content": [{"type": "text", "text": prompt}] + image_content}
    ]

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages
        }
    }


def load_checkpoint(checkpoint_path):
    """Load checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"processed_videos": [], "pseudo_labels": []}


def save_checkpoint(checkpoint_path, processed_videos, pseudo_labels):
    """Save checkpoint for resume capability."""
    with open(checkpoint_path, 'w') as f:
        json.dump({"processed_videos": processed_videos, "pseudo_labels": pseudo_labels}, f)


def save_pseudo_labels(pseudo_labels, output_path):
    """Save pseudo labels in the same format as original data."""
    with open(output_path, 'w') as f:
        json.dump(pseudo_labels, f, indent=4)
    print(f"Saved {len(pseudo_labels)} pseudo labels to {output_path}")


def prepare_video_requests(video_ids, video_captions, args):
    """Prepare batch requests for all videos.
    
    Returns:
        List of (video_id, request_dict, num_to_generate) tuples
    """
    requests = []
    
    for video_id in tqdm(video_ids, desc="Preparing video requests"):
        try:
            existing_captions = video_captions[video_id]
            num_to_generate = max(0, args.target_captions - len(existing_captions))

            if num_to_generate == 0:
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
                print(f"Video not found: {video_id}")
                continue

            # Sample and encode frames
            frames = sample_video_frames(video_path, args.num_frames)
            if frames is None:
                raise ValueError(f"Failed to sample frames from video {video_path}")

            frames_b64 = encode_frames_base64(frames)
            prompt = build_gpt_prompt(existing_captions, num_to_generate)
            
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


def submit_batch(client, batch_file_path, description):
    """Submit batch file to OpenAI and return batch object."""
    with open(batch_file_path, 'rb') as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description}
    )
    
    print(f"Batch submitted: {batch.id}")
    print(f"Status: {batch.status}")
    return batch


def poll_batch_completion(client, batch_id, poll_interval=60):
    """Poll batch status until completion."""
    print(f"Polling batch {batch_id} (interval: {poll_interval}s)...")
    
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        print(f"Status: {status} | Completed: {batch.request_counts.completed}/{batch.request_counts.total} | "
              f"Failed: {batch.request_counts.failed}")
        
        if status == "completed":
            print("Batch completed successfully!")
            return batch
        elif status == "failed" or status == "expired" or status == "cancelled":
            print(f"Batch {status}!")
            return batch
        
        time.sleep(poll_interval)


def parse_batch_results(client, batch, requests_info):
    """Parse batch results and extract captions.
    
    Args:
        client: OpenAI client
        batch: Completed batch object
        requests_info: List of (video_id, request, num_to_generate) tuples
        
    Returns:
        List of pseudo label dicts
    """
    # Create mapping from video_id to expected count
    video_to_count = {video_id: num_gen for video_id, _, num_gen in requests_info}
    
    # Download results
    result_file_id = batch.output_file_id
    if not result_file_id:
        print("No output file available!")
        return []
    
    result_content = client.files.content(result_file_id)
    result_lines = result_content.text.strip().split('\n')
    
    pseudo_labels = []
    successful_count = 0
    failed_count = 0
    
    for line in result_lines:
        try:
            result = json.loads(line)
            video_id = result['custom_id']
            expected_count = video_to_count.get(video_id, 0)
            
            if result.get('error'):
                print(f"Error for {video_id}: {result['error']}")
                failed_count += 1
                continue
            
            # Extract content
            content = result['response']['body']['choices'][0]['message']['content'].strip()
            
            # Clean markdown formatting
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            # Parse captions
            captions = json.loads(content)
            
            # Validate
            if not isinstance(captions, list):
                print(f"Invalid format for {video_id}: expected list, got {type(captions)}")
                failed_count += 1
                continue
            
            if len(captions) != expected_count:
                print(f"Caption count mismatch for {video_id}: expected {expected_count}, got {len(captions)}")
                # Still use the captions we got
            
            # Add to pseudo labels
            for caption in captions:
                pseudo_labels.append({"video": video_id, "caption": caption})
            
            successful_count += 1
            
        except Exception as e:
            print(f"Error parsing result: {e}")
            failed_count += 1
            continue
    
    print(f"Successfully parsed: {successful_count}/{len(result_lines)}")
    print(f"Failed: {failed_count}/{len(result_lines)}")
    
    return pseudo_labels


def parse_args():
    parser = argparse.ArgumentParser(description='Generate pseudo labels using GPT Batch API')
    parser.add_argument('--dataset', type=str, default='didemo', choices=['msrvtt', 'didemo', 'actnet', 'lsmdc'],
                        help='Dataset to process')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root data directory')
    parser.add_argument('--video_dir', type=str, default='./data_process/datasets/DiDeMo/train/videos',
                        help='Directory containing video files')
    parser.add_argument('--target_captions', type=int, default=50,
                        help='Target total captions per video')
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames to sample per video')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start processing from this video index')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='End processing at this video index (-1 for all)')
    parser.add_argument('--poll_interval', type=int, default=60,
                        help='Batch status polling interval in seconds (default: 60)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Maximum requests per batch')
    parser.add_argument('--model', type=str, default='gpt-5-mini-2025-08-07',
                        help='OpenAI model')
    return parser.parse_args()


def main():
    """Main function using OpenAI Batch API for cost-efficient processing."""
    args = parse_args()

    # Verify API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Setup paths
    data_path = os.path.join(args.data_dir, args.dataset, 'video_retreival_caption', f'{args.dataset}_ret_train.json')
    output_path = os.path.join(args.data_dir, args.dataset, 'video_retreival_caption', f'{args.dataset}_ret_train_pesudo.json')
    batch_dir = os.path.join(args.data_dir, args.dataset, 'video_retreival_caption', 'batch_files')
    os.makedirs(batch_dir, exist_ok=True)
    
    # Load existing pseudo labels if resuming
    all_pseudo_labels = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            all_pseudo_labels = json.load(f)
        print(f"Loaded {len(all_pseudo_labels)} existing pseudo labels from {output_path}")

    print(f"{'='*60}")
    print(f"OpenAI Batch API Pseudo Label Generation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Loading data from: {data_path}")
    print(f"Video directory: {args.video_dir}")
    print(f"Output path: {output_path}")
    print(f"Batch directory: {batch_dir}")

    # Load data
    video_captions = load_train_data(data_path)
    video_ids = sorted(video_captions.keys())

    # Apply index range
    end_idx = args.end_idx if args.end_idx > 0 else len(video_ids)
    video_ids = video_ids[args.start_idx:end_idx]

    print(f"\nTotal videos to process: {len(video_ids)}")
    print(f"Target captions per video: {args.target_captions}")
    print(f"Max batch size: {args.batch_size} requests")

    # Step 1: Prepare batch requests
    print(f"\n{'='*60}")
    print(f"Step 1: Preparing batch requests")
    print(f"{'='*60}")
    start_time = time.time()
    
    requests_info = prepare_video_requests(video_ids, video_captions, args)
    
    if len(requests_info) == 0:
        print("No videos to process!")
        return
    
    prep_time = time.time() - start_time
    print(f"Prepared {len(requests_info)} requests in {prep_time:.1f}s")

    # Split into batches if needed (API limit is 50k requests per batch)
    num_batches = (len(requests_info) + args.batch_size - 1) // args.batch_size
    print(f"Will submit {num_batches} batch(es) to OpenAI")
    
    # Process each batch
    for batch_idx in range(num_batches):
        batch_start_idx = batch_idx * args.batch_size
        batch_end_idx = min((batch_idx + 1) * args.batch_size, len(requests_info))
        batch_requests = requests_info[batch_start_idx:batch_end_idx]
        
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_idx + 1}/{num_batches}")
        print(f"Requests: {len(batch_requests)}")
        print(f"{'='*60}")

        # Step 2: Create batch file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_file_path = os.path.join(batch_dir, f'batch_{timestamp}_{batch_idx}.jsonl')
        create_batch_file(batch_requests, batch_file_path)

        # Step 3: Submit batch
        print(f"\nSubmitting batch to OpenAI...")
        batch_description = f"{args.dataset}_pseudo_labels_batch_{batch_idx}"
        batch = submit_batch(client, batch_file_path, batch_description)

        # Step 4: Poll for completion
        print(f"\nWaiting for batch completion...")
        completed_batch = poll_batch_completion(client, batch.id, args.poll_interval)

        if completed_batch.status != "completed":
            print(f"Warning: Batch {batch_idx + 1} did not complete successfully!")
            continue

        # Step 5: Parse results
        print(f"\nParsing batch results...")
        pseudo_labels = parse_batch_results(client, completed_batch, batch_requests)
        all_pseudo_labels.extend(pseudo_labels)
        
        print(f"Batch {batch_idx + 1} generated {len(pseudo_labels)} pseudo labels")
        
        # Step 6: Save incrementally after each batch for stability
        print(f"Saving progress after batch {batch_idx + 1}...")
        save_pseudo_labels(all_pseudo_labels, output_path)
        print(f"Current total: {len(all_pseudo_labels)} pseudo labels saved")

    # Final summary
    print(f"\n{'='*60}")
    print(f"All batches completed")
    print(f"{'='*60}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Total videos processed: {len(requests_info)}")
    print(f"Total pseudo labels generated: {len(all_pseudo_labels)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Output saved to: {output_path}")
    print(f"\nCost savings: ~50% compared to standard API!")


if __name__ == "__main__":
    main()
