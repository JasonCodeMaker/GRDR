import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import (
    VideoTextGuidedDataset,
    MultiTextVideoDataset,
    load_internvideo2_features,
)
from .models.videorqvae import VideoRQVAE_V2

# VideoRQVAE_V2 only works with InternVideo2 pooled embeddings
FEATURE_EXTRACTOR = "InternVideo2"

PREFIX = [
    "A_{}", "B_{}", "C_{}", "D_{}", "E_{}", "F_{}", "G_{}", "H_{}",
    "I_{}", "J_{}", "K_{}", "L_{}", "M_{}", "N_{}", "O_{}", "P_{}",
    "Q_{}", "R_{}", "S_{}", "T_{}", "U_{}", "V_{}", "W_{}", "X_{}",
    "Y_{}", "Z_{}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoRQVAE_V2 Semantic ID Generation (InternVideo2 only)")

    parser.add_argument('--version', type=str, default="2.0", help="version of the model")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="index/log/msrvtt/videorqvae_v2.0/tokens_4_codes_256_layers_4/0108_1624/best_recall_at_1_model.pth",
        help="Path to trained VideoRQVAE_V2 model checkpoint directory",
    )
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for generated indices")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")

    parser.add_argument("--dataset", type=str, default="msrvtt", choices=["msrvtt", "didemo", "actnet", "activitynet", "lsmdc"])
    parser.add_argument(
        "--features_root",
        type=str,
        default="./dataset/features",
        help="Path to features directory",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])

    parser.add_argument(
        "--extract_codebook",
        action="store_true",
        default=True,
        help="Extract and save codebook embeddings as codebook_embedding.pt",
    )
    parser.add_argument(
        "--codebook_output_path",
        type=str,
        default=None,
        help="Output path for codebook_embedding.pt (default: same as output_dir)",
    )

    return parser.parse_args()


def collate_variable_text_batch(batch):
    """Custom collate function to handle variable number of texts per video.

    Pads text_embs to max_texts in batch and creates a mask for valid entries.
    This is necessary for datasets like ActivityNet where videos have different
    numbers of captions (e.g., 2-4), unlike MSRVTT which has exactly 20 per video.
    """
    # Find max number of texts in this batch
    max_texts = max(item['text_embs'].shape[0] for item in batch)
    text_dim = batch[0]['text_embs'].shape[-1]
    batch_size = len(batch)

    # Initialize padded tensors
    video_patches = torch.stack([item['video_patches'] for item in batch])
    text_embs_padded = torch.zeros(batch_size, max_texts, text_dim)
    text_masks = torch.zeros(batch_size, max_texts, dtype=torch.bool)

    # Pad text_group_ids if present
    has_group_ids = batch[0]['text_group_ids'] is not None
    text_group_ids_padded = None
    if has_group_ids:
        text_group_ids_padded = torch.zeros(batch_size, max_texts, dtype=torch.long)

    # Fill in the data
    for i, item in enumerate(batch):
        num_texts = item['text_embs'].shape[0]
        text_embs_padded[i, :num_texts] = item['text_embs']
        text_masks[i, :num_texts] = True

        if has_group_ids:
            text_group_ids_padded[i, :num_texts] = item['text_group_ids']

    return {
        'video_patches': video_patches,
        'text_embs': text_embs_padded,
        'text_masks': text_masks,  # [batch_size, max_texts] - True for valid, False for padding
        'video_id': [item['video_id'] for item in batch],
        'text_keys': [item['text_keys'] for item in batch],
        'text_group_ids': text_group_ids_padded
    }


def extract_codebook_embeddings(model: VideoRQVAE_V2, expected_layers: int, expected_size: int, expected_dim: int) -> torch.Tensor:
    if not hasattr(model, "rq"):
        raise RuntimeError("Model missing residual quantizer (rq)")
    if not hasattr(model.rq, "get_codebook"):
        raise RuntimeError("Residual quantizer has no get_codebook method")

    codebook = model.rq.get_codebook()  # (layers, size, dim)
    layers, size, dim = codebook.shape
    if layers != expected_layers:
        print(f"Warning: codebook has {layers} layers (expected {expected_layers})")
    if size != expected_size:
        print(f"Warning: codebook size {size} (expected {expected_size})")
    if dim != expected_dim:
        print(f"Warning: codebook dim {dim} (expected {expected_dim})")

    flattened = codebook.reshape(layers * size, dim).contiguous()
    if torch.isnan(flattened).any() or torch.isinf(flattened).any():
        raise RuntimeError("Codebook contains NaN or Inf values")
    return flattened


def prepare_dataset(
    args: argparse.Namespace,
    num_latent_tokens: int,
    train_video_features,
    train_text_features,
    test_video_features,
):
    """Prepare dataset for VideoRQVAE_V2 (InternVideo2 pooled embeddings)."""
    if args.split == "train":
        # MultiTextVideoDataset returns all captions per video with k-means text_group_ids
        dataset = MultiTextVideoDataset(
            args.dataset,
            args.features_root,
            split="train",
            feature_extractor=FEATURE_EXTRACTOR,
            video_features=train_video_features,
            text_features=train_text_features,
            num_latent_tokens=num_latent_tokens,  # Required for k-means clustering
        )
        if not len(dataset.video_ids):
            raise ValueError(f"No videos found in {args.dataset} train split")

        # InternVideo2 features are pooled: [in_dim] instead of [num_patches, in_dim]
        sample = dataset.video_text_groups[dataset.video_ids[0]]
        video_feature = sample["video"]
        if isinstance(video_feature, torch.Tensor):
            dim = video_feature.shape[-1] if video_feature.ndim > 0 else video_feature.numel()
        else:
            dim = len(video_feature)

        dataset.num_patches = 1  # Pooled embedding
        dataset.dim = dim
        print(f"Loaded TRAIN dataset: {len(dataset)} videos, pooled {dim}D embeddings")
        return dataset

    # Test split: standard video-text pairs
    dataset = VideoTextGuidedDataset(
        args.dataset,
        args.features_root,
        split="test",
        text_guided=False,
        model_type="videorqvae",
        feature_extractor=FEATURE_EXTRACTOR,
        video_features=test_video_features,
        text_features=None,
    )
    if not len(dataset):
        raise ValueError(f"No videos found in {args.dataset} test split")

    # InternVideo2 features are pooled
    sample = dataset[0]
    video_tensor = sample.get("video_patches") if isinstance(sample, dict) else sample
    if isinstance(video_tensor, torch.Tensor):
        dim = video_tensor.shape[-1] if video_tensor.ndim > 0 else video_tensor.numel()
    else:
        dim = len(video_tensor)

    dataset.num_patches = 1
    dataset.dim = dim
    print(f"Loaded TEST dataset: {len(dataset)} videos, pooled {dim}D embeddings")
    return dataset


def load_model_from_checkpoint(ckpt: dict, dataset) -> VideoRQVAE_V2:
    """
    Load VideoRQVAE_V2 from checkpoint (model_config format only).

    Args:
        ckpt: Checkpoint dict containing 'model_config'
        dataset: Dataset instance (used to override in_dim)

    Returns:
        VideoRQVAE_V2 model instance
    """
    if 'model_config' not in ckpt:
        raise ValueError("Checkpoint missing 'model_config'. Only VideoRQVAE_V2 checkpoints are supported.")

    print("Loading VideoRQVAE_V2 from checkpoint (model_config)")
    return VideoRQVAE_V2.from_config(
        ckpt['model_config'],
        in_dim=dataset.dim,
    )

def save_codebook(
    model: VideoRQVAE_V2,
    num_emb_list,
    e_dim: int,
    output_dir: str,
):
    codebook = extract_codebook_embeddings(
        model=model,
        expected_layers=len(num_emb_list),
        expected_size=num_emb_list[0],
        expected_dim=e_dim,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "codebook_embedding.pt")
    torch.save(codebook.cpu(), output_path)
    print(f"Saved codebook embeddings to {output_path} (shape {tuple(codebook.shape)})")


def format_semantic_id(indices):
    formatted = []
    for idx, code in enumerate(indices):
        if idx < len(PREFIX):
            formatted.append(PREFIX[idx].format(int(code)))
        else:
            formatted.append(f"L{idx}_{int(code)}")
    return formatted

def process_train_batches(model: VideoRQVAE_V2, data_loader, device: torch.device) -> dict:
    """
    Process training batches using k-means group assignments for multi-text semantic ID generation.

    VideoRQVAE_V2 uses pre-computed k-means clustering to assign each text to a latent token.
    Each text's group_id (0 to num_latent_tokens-1) determines which token's codes to use.

    Args:
        model: Trained VideoRQVAE_V2 model
        data_loader: DataLoader returning MultiTextVideoDataset batches with text_group_ids
        device: Device for inference

    Returns:
        dict: {video_id: {query_id: semantic_id_list}} - Semantic IDs per query
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing train videos (k-means based)"):
            video_patches = batch["video_patches"].to(device)  # [batch_size, in_dim] - Pooled InternVideo2 features
            text_group_ids = batch["text_group_ids"]           # [batch_size, max_texts] - K-means assignments
            text_masks = batch["text_masks"]                   # [batch_size, max_texts] - True for valid texts
            video_ids = batch["video_id"]                      # [batch_size]
            text_keys = batch["text_keys"]                     # [batch_size][num_texts] - list of lists

            # Get quantization indices for all latent tokens
            indices, _ = model.get_indices(video_patches, use_sk=False)
            indices = indices.cpu().numpy()  # [batch_size, num_latent_tokens, num_layers]

            batch_size = len(video_ids)

            # Extract semantic IDs using k-means group assignments
            for b_idx in range(batch_size):
                video_id = video_ids[b_idx]
                query_map = {}
                video_text_keys = text_keys[b_idx]  # Get text keys for this video

                for text_idx, text_key in enumerate(video_text_keys):
                    # Skip padding texts (only for batches with variable text counts)
                    if text_masks is not None and not text_masks[b_idx, text_idx]:
                        continue

                    # Get k-means group assignment (which token to use)
                    group_id = int(text_group_ids[b_idx, text_idx])

                    # Extract semantic ID from the assigned token
                    token_codes = indices[b_idx, group_id]  # [num_layers]
                    semantic_id = [int(x) for x in token_codes]

                    # Generate query_id same way as original
                    query_id = text_key.split("_")[-1] if "_" in text_key else str(text_idx)
                    query_map[query_id] = semantic_id

                results[video_id] = query_map

    return results


def process_test_batches(model: VideoRQVAE_V2, data_loader, device: torch.device) -> dict:
    """
    Process test batches to generate semantic IDs for all videos.

    Args:
        model: Trained VideoRQVAE_V2 model
        data_loader: DataLoader returning structured test data
        device: Device for inference

    Returns:
        dict: {video_id: [[token_0_codes], [token_1_codes], ...]} - semantic IDs for all latent tokens
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing test videos"):
            video_patches = batch['video_patches'].to(device)  # [batch_size, in_dim] - Pooled InternVideo2 features
            video_ids = batch['video_id']  # [batch_size] - list of video IDs

            # Get quantization indices for all latent tokens
            indices, _ = model.get_indices(video_patches, use_sk=False)
            indices = indices.cpu().numpy()  # [batch_size, num_latent_tokens, num_layers]

            # Map each video ID to its semantic ID codes for all tokens
            for b_idx in range(len(video_ids)):
                video_id = video_ids[b_idx]

                # Extract semantic IDs for all latent tokens of this video
                token_codes = [
                    [int(x) for x in indices[b_idx, token_idx]]  # Convert each token's codes
                    for token_idx in range(indices.shape[1])  # Iterate over num_latent_tokens
                ]
                results[video_id] = token_codes

    return results


def format_train_output(raw: dict, dataset_name: str = None) -> dict:
    """
    Optimized format training output using video-centric processing.

    Performance improvement: O(n) complexity instead of O(n²) by eliminating
    the expensive caption counting operation.

    Args:
        raw: Raw semantic IDs dict from process_train_batches
        dataset_name: Required for unified format, e.g., 'msrvtt'

    Returns:
        List of unified format entries [{"video": "...", "caption": "...", "SemanticID": [...]}]
    """
    # Load original training captions
    if dataset_name == "activitynet":
        original_data_path = f"./data/actnet/video_retreival_caption/actnet_ret_train.json"
    else:
        original_data_path = f"./data/{dataset_name}/video_retreival_caption/{dataset_name}_ret_train.json"

    print(f"Loading original captions from {original_data_path}")
    try:
        with open(original_data_path, "r") as f:
            original_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Original caption file not found: {original_data_path}")

    print(f"Loaded {len(original_data)} original video-caption pairs")

    # Convert raw semantic IDs to formatted version
    formatted_semantic_ids = {
        video_id: {query_id: format_semantic_id(indices) for query_id, indices in queries.items()}
        for video_id, queries in raw.items()
    }

    # Step 1: Determine video processing order (maintains original order)
    seen_videos = set()
    video_order = []
    for item in original_data:
        video_filename = item["video"]
        if video_filename not in seen_videos:
            video_order.append(video_filename)
            seen_videos.add(video_filename)

    print(f"Processing {len(video_order)} unique videos in original order")

    # Step 2: Group captions by video (single O(n) pass)
    video_captions = {}
    for item in original_data:
        video_filename = item["video"]
        if video_filename not in video_captions:
            video_captions[video_filename] = []
        video_captions[video_filename].append(item["caption"])

    # Step 3: Video-centric processing in original order
    unified_output = []
    matched_pairs = 0
    missing_semantic_ids = 0

    # Progress bar tracks videos instead of individual captions for efficiency
    pbar = tqdm(
        video_order,
        desc=f"Processing {dataset_name} videos",
        unit="videos"
    )

    for video_filename in pbar:
        # Extract video_id: remove directory prefix and file extension
        # LSMDC: "3001_21_JUMP_STREET/3001_21_JUMP_STREET_00.02.55.644-00.02.56.718.avi" -> "3001_21_JUMP_STREET_00.02.55.644-00.02.56.718"
        # MSRVTT: "video1234.mp4" -> "video1234"
        video_basename = os.path.basename(video_filename)  # Remove directory
        video_id = os.path.splitext(video_basename)[0]  # Remove extension
        captions = video_captions[video_filename]

        # Process all captions for this video together (eliminates O(n²) counting)
        for caption_idx, caption_data in enumerate(captions):
            # Check if caption_data is a list (ActivityNet/DiDeMo) or string (MSRVTT)
            if isinstance(caption_data, list):
                # ActivityNet/DiDeMo case: expand list into separate entries
                for sub_idx, single_caption in enumerate(caption_data):
                    query_id = str(sub_idx)
                    
                    # Get semantic ID if available
                    semantic_id = []
                    if video_id in formatted_semantic_ids and query_id in formatted_semantic_ids[video_id]:
                        semantic_id = formatted_semantic_ids[video_id][query_id]
                        matched_pairs += 1
                    else:
                        missing_semantic_ids += 1
                    
                    # Create unified entry with single caption string
                    unified_entry = {
                        "video": video_filename,
                        "caption": single_caption,
                        "SemanticID": semantic_id
                    }
                    unified_output.append(unified_entry)
            else:
                # MSRVTT case: caption is already a string
                query_id = str(caption_idx)  # Direct index, no counting needed

                # Get semantic ID if available
                semantic_id = []
                if video_id in formatted_semantic_ids and query_id in formatted_semantic_ids[video_id]:
                    semantic_id = formatted_semantic_ids[video_id][query_id]
                    matched_pairs += 1
                else:
                    missing_semantic_ids += 1

                # Create unified entry
                unified_entry = {
                    "video": video_filename,
                    "caption": caption_data,
                    "SemanticID": semantic_id
                }
                unified_output.append(unified_entry)

    pbar.close()

    print(f"Generated unified output with {len(unified_output)} video-caption pairs")
    print(f"Successfully matched semantic IDs: {matched_pairs}")
    print(f"Missing semantic IDs: {missing_semantic_ids}")

    return unified_output


def format_test_output(raw: dict) -> dict:
    return {
        video_id: [format_semantic_id(indices) for indices in semantic_ids]
        for video_id, semantic_ids in raw.items()
    }


def report_train_summary(output_dict: dict):
    total_queries = sum(len(queries) for queries in output_dict.values())
    num_videos = len(output_dict)
    avg_queries = total_queries / num_videos if num_videos else 0.0
    print(f"Total query-semantic pairs: {total_queries}")
    print(f"Average queries per video: {avg_queries:.1f}")
    if output_dict:
        sample_video, sample_queries = next(iter(output_dict.items()))
        sample_query, sample_sid = next(iter(sample_queries.items()))
        print(f"Sample [{sample_video} → {sample_query}]: {sample_sid[:3]}... ({len(sample_sid)} tokens)")


def report_test_summary(output_dict: dict):
    total_ids = sum(len(ids) for ids in output_dict.values())
    num_videos = len(output_dict)
    avg_ids = total_ids / num_videos if num_videos else 0.0
    print(f"Total semantic IDs: {total_ids}")
    print(f"Average semantic IDs per video: {avg_ids:.1f}")
    if output_dict:
        sample_video, sample_ids = next(iter(output_dict.items()))
        if sample_ids:
            print(f"Sample [{sample_video}]: {sample_ids[0][:3]}... ({len(sample_ids[0])} tokens)")


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading VideoRQVAE_V2 checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)

    # Print checkpoint metrics
    best_loss = ckpt.get("best_loss")
    if best_loss is not None:
        print(f"Best loss: {best_loss:.6f}")
    best_collision_rate = ckpt.get("best_collision_rate")
    if best_collision_rate is not None:
        print(f"Best collision rate: {best_collision_rate:.6f}")
    best_test_recon_loss = ckpt.get("best_test_recon_loss")
    if best_test_recon_loss is not None:
        print(f"Best test recon loss: {best_test_recon_loss:.6f}")

    # Extract model configuration
    if 'model_config' not in ckpt:
        raise ValueError("Checkpoint missing 'model_config'. Only VideoRQVAE_V2 checkpoints are supported.")

    model_config = ckpt['model_config']
    num_latent_tokens = model_config.get('num_latent_tokens', 4)

    # Load InternVideo2 features using shared function from datasets.py
    print(f"Loading {FEATURE_EXTRACTOR} features for {args.dataset}...")
    train_video_features, train_text_features, test_video_features, _ = load_internvideo2_features(
        args.dataset, args.features_root
    )

    # Prepare dataset (MultiTextVideoDataset for train with k-means, VideoTextGuidedDataset for test)
    dataset = prepare_dataset(
        args,
        num_latent_tokens,
        train_video_features,
        train_text_features,
        test_video_features,
    )

    # Load VideoRQVAE_V2 model
    model = load_model_from_checkpoint(ckpt, dataset)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded on {device} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    # Extract config parameters for output path construction
    num_emb_list = model_config['num_emb_list']
    code_num = num_emb_list[0] if num_emb_list else 128
    codebook_layers = len(num_emb_list) if num_emb_list else 3

    output_root = os.path.join(
        args.output_dir,
        f"{args.dataset}/none/videorqvae_v{args.version}_c{code_num}_l{codebook_layers}",
    )
    os.makedirs(output_root, exist_ok=True)

    # Extract and save codebook embeddings if requested
    if args.extract_codebook:
        save_codebook(model, num_emb_list, model_config['e_dim'], args.codebook_output_path or output_root)

    # Create DataLoader with appropriate collate function
    num_workers = 4  # Default for VideoRQVAE_V2
    collate_fn = collate_variable_text_batch if args.split == "train" else None
    data_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    # Generate semantic IDs
    if args.split == "train":
        # Process train split using k-means group assignments
        raw = process_train_batches(model, data_loader, device)
        # Generate unified format directly
        unified_output_file = os.path.join(output_root, f"{args.dataset}_videorqvae_index_{args.split}.json")
        formatted = format_train_output(raw, dataset_name=args.dataset)
        with open(unified_output_file, "w") as handle:
            json.dump(formatted, handle, indent=4)
        print(f"Unified semantic IDs written to {unified_output_file}")

    else:
        # Handle test split - generate codes for all tokens
        output_file = os.path.join(output_root, f"{args.dataset}_videorqvae_index_{args.split}.json")
        raw = process_test_batches(model, data_loader, device)
        formatted = format_test_output(raw)
        with open(output_file, "w") as handle:
            json.dump(formatted, handle, indent=4)
        print(f"Semantic IDs written to {output_file}")
        report_test_summary(formatted)

if __name__ == "__main__":
    main()
