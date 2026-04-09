import argparse
import json
import os
import pickle
from time import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import VideoTextGuidedDataset, MultiTextVideoDataset
from .models.rqvae import VideoRQVAE, VideoRQVAE_V2

FEATURE_CONFIG = {
    "CLIP": ("CLIP", "cliplargel14"),
    "InternVL": ("InternVL", "internvl-hico-r16"),
    "InternVideo2": ("InternVideo2", "internvideo2"),
}

PREFIX = [
    "A_{}", "B_{}", "C_{}", "D_{}", "E_{}", "F_{}", "G_{}", "H_{}",
    "I_{}", "J_{}", "K_{}", "L_{}", "M_{}", "N_{}", "O_{}", "P_{}",
    "Q_{}", "R_{}", "S_{}", "T_{}", "U_{}", "V_{}", "W_{}", "X_{}",
    "Y_{}", "Z_{}",
]

PARAM_MAPPINGS = {
    "vq_beta": "beta",
    "quantization_beta": "beta",
    "codebook_beta": "beta",
    "diversity_weight": "diversity_loss_weight",
    "router_diversity_weight": "diversity_loss_weight",
    "num_codes": "code_num",
    "codebook_size": "code_num",
    "quantizer_layers": "codebook_layers",
    "num_quantizer_layers": "codebook_layers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoRQVAE Semantic ID Generation")

    parser.add_argument('--version', type=str, default="2.7", help="version of the model")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="index/log/msrvtt_none_v4.0/lr_0.005_vr_0.03_n_4_cn_256_cl_4_beta_0.35_dl_9.0/1102_14",
        help="Path to trained VideoRQVAE model checkpoint directory",
    )
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for generated indices")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")

    parser.add_argument("--dataset", type=str, default="msrvtt", choices=["msrvtt", "didemo", "actnet", "activitynet", "lsmdc"])
    parser.add_argument(
        "--features_root",
        type=str,
        default="./data_process/datasets/features",
        help="Path to features directory",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--type", type=str, default="text_guided", choices=["standard", "text_guided"])
    parser.add_argument("--frames", type=int, default=8, help="Number of frames used in feature extraction")
    parser.add_argument("--feature_extractor", type=str, default="InternVideo2", choices=["CLIP", "InternVL", "InternVideo2"])

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


def _load_pickle(path: str, required: bool = False, label: str = None):
    if not os.path.exists(path):
        message = f"{label or os.path.basename(path)} not found at {path}"
        if required:
            raise FileNotFoundError(message)
        print(message)
        return None

    start = time()
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    elapsed = time() - start
    size = f"{len(data)} entries" if hasattr(data, "__len__") else "loaded"
    print(f"Loaded {label or os.path.basename(path)} ({size}) in {elapsed:.2f}s from {path}")
    return data


def load_video_text_features(dataset: str, root: str, feature_extractor: str, frames: int):
    subdir, suffix = FEATURE_CONFIG[feature_extractor]
    base_dir = os.path.abspath(os.path.join(root, subdir))

    if feature_extractor == "InternVideo2":
        frame_dir = os.path.join(base_dir, "video")
    else:
        frame_dir = os.path.join(base_dir, str(frames))

    train_video = _load_pickle(
        os.path.join(frame_dir, f"{dataset}_{suffix}_video_embeddings_train.pkl"),
        required=True,
        label="train video features",
    )
    train_text = _load_pickle(
        os.path.join(base_dir, f"{dataset}_{suffix}_text_embeddings_train.pkl"),
        label="train text features",
    )
    test_video = _load_pickle(
        os.path.join(frame_dir, f"{dataset}_{suffix}_video_embeddings_test.pkl"),
        label="test video features",
    )
    return train_video, train_text, test_video


def normalize_model_args(raw_args) -> SimpleNamespace:
    args_dict = vars(raw_args).copy() if hasattr(raw_args, "__dict__") else dict(raw_args)
    for old_name, new_name in PARAM_MAPPINGS.items():
        if old_name in args_dict and new_name not in args_dict:
            args_dict[new_name] = args_dict[old_name]
    return SimpleNamespace(**args_dict)


def extract_codebook_embeddings(model: VideoRQVAE, expected_layers: int, expected_size: int, expected_dim: int) -> torch.Tensor:
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


def _infer_video_dims(video):
    tensor = video if isinstance(video, torch.Tensor) else torch.as_tensor(video)
    if tensor.ndim == 1:
        return 1, int(tensor.shape[0])
    return int(tensor.shape[0]), int(tensor.shape[1])


def prepare_dataset(
    args: argparse.Namespace,
    feature_extractor: str,
    frames: int,
    train_video_features,
    train_text_features,
    test_video_features,
):
    if args.split == "train":
        dataset = MultiTextVideoDataset(
            args.dataset,
            args.features_root,
            split="train",
            feature_extractor=feature_extractor,
            video_features=train_video_features,
            text_features=train_text_features,
        )
        if not len(dataset.video_ids):
            raise ValueError(f"No videos found in {args.dataset} train split")
        sample = dataset.video_text_groups[dataset.video_ids[0]]
        num_patches, dim = _infer_video_dims(sample["video"])
        dataset.num_patches = num_patches
        dataset.dim = dim
        print(f"Loaded TRAIN dataset: {len(dataset)} videos, {num_patches} patches × {dim}D")
        return dataset

    dataset = VideoTextGuidedDataset(
        args.dataset,
        args.features_root,
        split="test",
        text_guided=False,
        model_type="videorqvae",
        feature_extractor=feature_extractor,
        video_features=test_video_features,
        text_features=None,
    )
    if not len(dataset):
        raise ValueError(f"No videos found in {args.dataset} test split")

    if not hasattr(dataset, "num_patches") or not hasattr(dataset, "dim"):
        sample = dataset[0]
        video_tensor = sample.get("video_patches") if isinstance(sample, dict) else sample
        num_patches, dim = _infer_video_dims(video_tensor)
        dataset.num_patches = num_patches
        dataset.dim = dim
    print(f"Loaded TEST dataset: {len(dataset)} videos, {dataset.num_patches} patches × {dataset.dim}D")
    return dataset


def load_model_from_checkpoint(ckpt: dict, dataset, feature_extractor: str) -> VideoRQVAE:
    """
    Load VideoRQVAE from checkpoint with automatic fallback for old checkpoints.

    New checkpoints (saved with model_config): Simple one-line loading
    Old checkpoints (saved with args): Fallback to parameter extraction

    Args:
        ckpt: Checkpoint dict containing either 'model_config' or 'args'
        dataset: Dataset instance (used to override in_dim/num_patches)

    Returns:
        VideoRQVAE model instance
    """
    # New format: Use model_config if available (clean path)
    if 'model_config' in ckpt:
        print("Loading model from new checkpoint format (model_config)")

        if feature_extractor == "InternVideo2":
            return VideoRQVAE_V2.from_config(
                ckpt['model_config'],
                in_dim=dataset.dim,
                num_patches=dataset.num_patches
            )
        else:
            return VideoRQVAE.from_config(
                ckpt['model_config'],
                in_dim=dataset.dim,
                num_patches=dataset.num_patches
            )

    # Backward compatibility: Fallback to args-based loading for old checkpoints
    print("Loading model from legacy checkpoint format (args) - consider re-training")
    raw_args = ckpt['args']
    model_args = normalize_model_args(raw_args)

    # Extract parameters from old checkpoint
    if getattr(model_args, "num_emb_list", None):
        num_emb_list = list(model_args.num_emb_list)
    elif hasattr(model_args, "code_num") and hasattr(model_args, "codebook_layers"):
        num_emb_list = [model_args.code_num] * model_args.codebook_layers
    else:
        raise ValueError("Checkpoint missing num_emb_list configuration")

    beta = model_args.beta
    diversity_loss_weight = model_args.diversity_loss_weight
    version = getattr(model_args, 'version', '1.1')

    # Reconstruct model using old parameter extraction logic
    return VideoRQVAE(
        in_dim=dataset.dim,
        num_patches=dataset.num_patches,
        num_latent_tokens=getattr(model_args, "num_latent_tokens"),
        encoder_width=getattr(model_args, "encoder_width"),
        encoder_layers=getattr(model_args, "encoder_layers"),
        encoder_heads=getattr(model_args, "encoder_heads"),
        num_emb_list=num_emb_list,
        e_dim=model_args.e_dim,
        dropout_prob=getattr(model_args, "dropout_prob", 0.0),
        bn=getattr(model_args, "bn", False),
        vid_loss_weight=getattr(model_args, "vid_loss_weight", [0.0, 0.0, 1.0, 0.0, 0.0]),
        text_loss_type=getattr(model_args, "text_loss_type", "contrastive"),
        text_loss_pos=getattr(model_args, "text_loss_pos", "after"),
        quant_loss_weight=getattr(model_args, "quant_loss_weight", 1.0),
        kmeans_init=getattr(model_args, "kmeans_init", True),
        kmeans_iters=getattr(model_args, "kmeans_iters", 100),
        sk_epsilons=getattr(model_args, "sk_epsilons", 0.0),
        sk_iters=getattr(model_args, "sk_iters", 50),
        use_linear=getattr(model_args, "use_linear", 0),
        beta=beta,
        use_ema=getattr(model_args, "use_ema", False),
        ema_decay=getattr(model_args, "ema_decay", 0.99),
        use_text_decoder=getattr(model_args, "use_text_decoder", False),
        text_dim=getattr(model_args, "text_dim", 4096),
        text_decoder_layers=getattr(model_args, "text_decoder_layers", [1536, 3072]),
        router_hidden_dim=getattr(model_args, "router_hidden_dim", 768),
        router_temperature=getattr(model_args, "router_temperature", 1.5),
        soft_temperature=getattr(model_args, "soft_temperature", 1.0),
        text_recon_loss_weight=getattr(model_args, "text_recon_loss_weight", 0.5),
        diversity_loss_weight=diversity_loss_weight,
        contrastive_temperature=getattr(model_args, "contrastive_temperature", 0.07),
        num_frames=getattr(model_args, "frames", 8),
        frame_temperature=getattr(model_args, "frame_temperature", 0.1),
        num_iterations=getattr(model_args, "num_iterations", 3),
        version=version,
        encoder=getattr(model_args, "encoder", "VideoLatentEncoder"),
        decoder=getattr(model_args, "decoder", "VideoLatentDecoder"),
    )

def save_codebook(
    model: VideoRQVAE,
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

def extract_semantic_ids_from_batch_forward(indices, selected_token_idx_all, video_ids, text_keys, feature_extractor: str):
    """
    Extract semantic IDs from batch forward results using the router-selected token indices.

    Args:
        indices: [batch_size, num_latent_tokens, num_layers] - All quantization indices
        selected_token_idx_all: List of [batch_size] tensors - Selected token indices for each text
        video_ids: [batch_size] - Video identifiers
        text_keys: [num_texts][batch_size] - Text key structure from dataset

    Returns:
        dict: {video_id: {query_id: semantic_id}} - Same format as original implementation
    """
    batch_size = len(video_ids)
    if feature_extractor == "InternVideo2":
        num_texts = selected_token_idx_all.shape[1]
    else:
        num_texts = len(selected_token_idx_all)

    results = {}

    for b_idx in range(batch_size):
        video_id = video_ids[b_idx]
        query_map = {}

        for text_idx in range(num_texts):
            # Get text key for this batch position and text index
            text_key = text_keys[text_idx][b_idx]

            # Get the selected token index from router output (directly from forward pass)
            if feature_extractor == "InternVideo2":
                selected_token_idx = selected_token_idx_all[b_idx][text_idx]
            else:
                selected_token_idx = selected_token_idx_all[text_idx][b_idx]

            # Extract semantic ID from the selected token
            token_codes = indices[b_idx, selected_token_idx]  # [num_layers]
            semantic_id = [int(x) for x in token_codes.cpu().numpy()]

            # Generate query_id same way as original
            query_id = text_key.split("_")[-1] if "_" in text_key else str(text_idx)
            query_map[query_id] = semantic_id

        results[video_id] = query_map

    return results


def process_train_batches(model: VideoRQVAE, data_loader, device: torch.device, feature_extractor: str) -> dict:
    """
    Optimized batch processing using model's forward pass for multi-text semantic ID generation.

    This version leverages the VideoRQVAE's forward method which processes all texts per video
    simultaneously, eliminating the nested loops and reducing forward passes from
    O(batch_size × num_texts) to O(1) per batch.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing train videos (batch optimized)"):
            video_patches = batch["video_patches"].to(device)  # [batch_size, num_patches, in_dim]
            text_embs = batch["text_embs"].to(device)          # [batch_size, num_texts, text_dim]
            video_ids = batch["video_id"]                      # [batch_size]
            text_keys = batch["text_keys"]                     # [num_texts][batch_size]

            # Single forward pass for entire batch - leverages model's multi-text processing
            # This processes all video-text pairs simultaneously instead of individual loops
            if feature_extractor == "InternVideo2":
                indices, selected_token_idx_all = model.get_indices(video_patches, text_embs=text_embs, return_semantic_selections=True)
            else:
                _, _, indices, _, _, _, selected_token_idx_all = model(
                    video_patches, text_embs, use_sk=False
                )

            # Extract semantic IDs using router selection results
            batch_results = extract_semantic_ids_from_batch_forward(
                indices, selected_token_idx_all, video_ids, text_keys, feature_extractor
            )

            # Merge batch results into final output
            results.update(batch_results)

    return results


def process_test_batches(model: VideoRQVAE, data_loader, device: torch.device, feature_extractor: str) -> dict:
    """
    Process test batches to generate semantic IDs for all videos.

    Args:
        model: Trained VideoRQVAE model
        data_loader: DataLoader returning structured test data
        device: Device for inference

    Returns:
        dict: {video_id: [[token_0_codes], [token_1_codes], ...]} - semantic IDs for all latent tokens
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing test videos"):
            video_patches = batch['video_patches'].to(device)  # [batch_size, num_patches, feature_dim]
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
        video_id = video_filename.replace(".mp4", "")
        captions = video_captions[video_filename]

        # Process all captions for this video together (eliminates O(n²) counting)
        for caption_idx, caption in enumerate(captions):
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
                "caption": caption,
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

    ckpt_file = os.path.join(args.ckpt_path, "best_test_loss_model.pth")
    print(f"Loading checkpoint from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

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

    # Extract feature extractor and frames info from checkpoint
    if 'model_config' in ckpt:
        # New format: Extract from model_config
        feature_extractor = ckpt.get('model_config', {}).get('feature_extractor') or args.feature_extractor
        frames = ckpt.get('model_config', {}).get('num_frames', args.frames)
    else:
        # Old format: Extract from args
        raw_model_args = ckpt["args"]
        model_args = normalize_model_args(raw_model_args)
        feature_extractor = model_args.feature_extractor
        frames = model_args.frames

    # Load features and prepare dataset
    train_video_features, train_text_features, test_video_features = load_video_text_features(
        args.dataset, args.features_root, feature_extractor=feature_extractor, frames=frames
    )

    dataset = prepare_dataset(
        args,
        feature_extractor,
        frames,
        train_video_features,
        train_text_features,
        test_video_features,
    )

    # Load model using new unified function (handles both old and new checkpoints)
    model = load_model_from_checkpoint(ckpt, dataset, feature_extractor)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded on {device} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    # Extract config parameters for output path construction
    model_config = model.get_config()
    num_emb_list = model_config['num_emb_list']
    code_num = num_emb_list[0] if num_emb_list else 256
    codebook_layers = len(num_emb_list) if num_emb_list else 4

    output_root = os.path.join(
        args.output_dir,
        f"{args.dataset}/none/videorqvae_v{args.version}_c{code_num}_l{codebook_layers}",
    )
    os.makedirs(output_root, exist_ok=True)

    if args.extract_codebook:
        save_codebook(model, num_emb_list, model_config['e_dim'], args.codebook_output_path or output_root)

    # Extract num_workers from checkpoint or use default
    if 'model_config' in ckpt:
        num_workers = 4  # Default for new checkpoints
    else:
        num_workers = getattr(normalize_model_args(ckpt["args"]), "num_workers", 4)

    data_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.split == "train":
        # Process semantic IDs using optimized batch processing
        # Performance improvement: ~10-20x faster by eliminating nested loops
        raw = process_train_batches(model, data_loader, device, feature_extractor)
        # Generate unified format directly
        unified_output_file = os.path.join(output_root, f"{args.dataset}_videorqvae_index_{args.split}.json")
        formatted = format_train_output(raw, dataset_name=args.dataset)
        with open(unified_output_file, "w") as handle:
            json.dump(formatted, handle, indent=4)
        print(f"Unified semantic IDs written to {unified_output_file}")

    else:
        # Handle test split
        output_file = os.path.join(output_root, f"{args.dataset}_videorqvae_index_{args.split}.json")
        raw = process_test_batches(model, data_loader, device, feature_extractor)
        formatted = format_test_output(raw)
        with open(output_file, "w") as handle:
            json.dump(formatted, handle, indent=4)
        print(f"Semantic IDs written to {output_file}")
        report_test_summary(formatted)

if __name__ == "__main__":
    main()
