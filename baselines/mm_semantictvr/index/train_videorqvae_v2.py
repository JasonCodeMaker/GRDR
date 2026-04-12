import argparse
import logging
import os
import pickle
from datetime import datetime
from time import time

import torch
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available - training metrics won't be logged to wandb")

from .datasets import MultiTextVideoDataset, VideoTextGuidedDataset, load_internvideo2_features
from .models.videorqvae import VideoRQVAE_V2
from .trainer_v2 import Trainer
from .utils import seed_everything, set_color, setup_logging_with_file

FEATURE_EXTRACTOR = "InternVideo2"
FEATURE_SUFFIX = "internvideo2"


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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VideoRQVAE_V2 on InternVideo2 pooled embeddings"
    )

    # Experiment setup
    parser.add_argument("--version", type=str, default="5.1")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--ckpt_dir", type=str, default="./index/log", help="checkpoint root directory")

    # Optimisation
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=5, help="validation frequency in epochs")

    parser.add_argument("--learner", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vq_lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-3)

    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)

    # Data
    parser.add_argument("--dataset", type=str, default="msrvtt", choices=["msrvtt", "didemo", "actnet", "activitynet", "lsmdc"])
    parser.add_argument("--features_root", type=str, default="./dataset/features")

    # Model
    parser.add_argument("--num_latent_tokens", type=int, default=4)
    parser.add_argument("--code_num", type=int, default=128)
    parser.add_argument("--codebook_layers", type=int, default=4)
    parser.add_argument("--e_dim", type=int, default=512)
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[1024, 768, 512], help='hidden sizes of encoder/decoder layers')
    parser.add_argument("--dropout_prob", type=float, default=0.15)
    parser.add_argument("--bn", action="store_true", help="enable batch norm inside encoder/decoder MLPs")
    parser.add_argument(
        "--vid_loss_weight",
        type=float,
        nargs="+",
        default=[1.0, 0.0, 1.0, 0.0, 0.0],
        help="Video loss weights [MSE, L1, Cosine, FrameCls, PatchNCE] (unused entries are ignored).",
    )
    parser.add_argument("--quant_loss_weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.35, help="commitment cost for RQ-VAE")
    parser.add_argument("--diversity_loss_weight", type=float, default=0.0)

    # Quantizer configuration
    parser.add_argument("--no_kmeans_init", dest="kmeans_init", action="store_false", default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--sk_epsilons", type=float, nargs="*", default=0.0, help="Sinkhorn epsilons per stage")
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.99)

    # Contrastive learning
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0, help="weight for contrastive loss in training")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false", default=True)
    parser.add_argument("--wandb_project", type=str, default="semantic-tvr")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()

def _log_model_configuration(logger, in_dim, args):
    logger.info(set_color("=" * 72, "blue"))
    logger.info(set_color("Model Configuration", "blue"))
    logger.info(set_color("=" * 72, "blue"))
    logger.info(f"Pooled feature dimension: {in_dim}")
    logger.info(f"Latent tokens: {args.num_latent_tokens}")
    logger.info(f"Embedding dimension (e_dim): {args.e_dim}")
    logger.info(f"MLP architecture: {args.mlp_layers}")
    logger.info(f"RQ codebooks: {args.codebook_layers} × {args.code_num}")
    logger.info(f"Video loss weights (MSE/L1/Cosine): {args.vid_loss_weight}")
    logger.info(f"Diversity loss weight: {args.diversity_loss_weight}")
    logger.info(f"Contrastive loss weight / temperature: {args.contrastive_loss_weight} / {args.contrastive_temperature}")
    logger.info(f"EMA updates: {'enabled' if args.use_ema else 'disabled'} (decay={args.ema_decay})")


def main():
    args = parse_args()

    # VideoRQVAE_V2 only works with InternVideo2 pooled embeddings
    args.feature_extractor = FEATURE_EXTRACTOR
    args.multi_text_mode = True
    args.text_guided = True
    args.use_text_decoder = False

    # Build checkpoint directory layout
    ckpt_root = os.path.abspath(args.ckpt_dir)
    exp_name = f"{args.dataset}/videorqvae_v{args.version}"
    hyp_suffix = (
        f"tokens_{args.num_latent_tokens}_codes_{args.code_num}_layers_{args.codebook_layers}")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    args.ckpt_dir = os.path.join(ckpt_root, exp_name, hyp_suffix, timestamp)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Configure logging early so data loading + training share the same file
    log_path = os.path.join(args.ckpt_dir, "train.log")
    logger = setup_logging_with_file(log_path, level=logging.INFO)
    logger.info(set_color("=" * 72, "blue"))
    logger.info(set_color("VideoRQVAE_V2 Training", "blue"))
    logger.info(set_color("=" * 72, "blue"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VideoRQVAE_V2 training.")
    torch.cuda.set_device(args.device)
    logger.info(f"Using CUDA device: {args.device}")

    seed_everything(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Pre-load feature dictionaries (saves time across worker processes)
    train_vid, train_txt, test_vid, test_txt = load_internvideo2_features(
        args.dataset, args.features_root
    )

    # Datasets: multi-text for train/val, single-text pairs for test evaluation
    logger.info(set_color("Building datasets...", "green"))
    train_dataset = MultiTextVideoDataset(
        args.dataset,
        args.features_root,
        split="train",
        feature_extractor=FEATURE_EXTRACTOR,
        video_features=train_vid,
        text_features=train_txt,
        num_latent_tokens=args.num_latent_tokens,
    )
    valid_dataset = MultiTextVideoDataset(
        args.dataset,
        args.features_root,
        split="train",
        feature_extractor=FEATURE_EXTRACTOR,
        video_features=train_vid,
        text_features=train_txt,
        num_latent_tokens=args.num_latent_tokens,
    )
    test_dataset = VideoTextGuidedDataset(
        args.dataset,
        args.features_root,
        split="test",
        text_guided=True,
        feature_extractor=FEATURE_EXTRACTOR,
        model_type="videorqvae",
        video_features=test_vid,
        text_features=test_txt,
    )

    logger.info(f"Train videos: {len(train_dataset)} | Valid videos: {len(valid_dataset)} | Test pairs: {len(test_dataset)}")

    # Inspect one training sample to derive tensor shapes
    sample = train_dataset[0]["video_patches"]
    if sample.dim() == 1:
        num_patches, in_dim = 1, sample.shape[0]
    else:
        num_patches, in_dim = sample.shape

    num_emb_list = [args.code_num] * args.codebook_layers
    model_config = {
        "in_dim": in_dim,
        "num_latent_tokens": args.num_latent_tokens,
        "num_emb_list": num_emb_list,
        "e_dim": args.e_dim,
        "mlp_layers": args.mlp_layers,
        "dropout_prob": args.dropout_prob,
        "bn": args.bn,
        "quant_loss_weight": args.quant_loss_weight,
        "kmeans_init": args.kmeans_init,
        "kmeans_iters": args.kmeans_iters,
        "sk_epsilons": args.sk_epsilons,
        "sk_iters": args.sk_iters,
        "use_linear": 0,  # Keep at 0 as per original design
        "beta": args.beta,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "diversity_loss_weight": args.diversity_loss_weight,
        "contrastive_temperature": args.contrastive_temperature,
        "vid_loss_weight": args.vid_loss_weight,
    }
    model = VideoRQVAE_V2(**model_config)
    model.config["feature_extractor"] = FEATURE_EXTRACTOR

    _log_model_configuration(logger, in_dim, args)
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_variable_text_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_variable_text_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"{args.dataset}_v{args.version}_n_{args.num_latent_tokens}_cn_{args.code_num}_cl_{args.codebook_layers}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        logger.info(set_color(f"wandb initialized: {args.wandb_project}/{run_name}", "green"))
    elif args.use_wandb and not WANDB_AVAILABLE:
        logger.warning("wandb requested but not installed; continuing without logging.")

    trainer = Trainer(args, model, video_key="video_patches", model_type="videorqvae")

    logger.info(set_color("=" * 72, "blue"))
    logger.info(set_color("Starting training", "blue"))
    logger.info(set_color("=" * 72, "blue"))

    best_loss, best_collision_rate = trainer.fit(train_loader, valid_loader, test_loader)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    logger.info(set_color("=" * 72, "green"))
    logger.info(set_color("Training complete", "green"))
    logger.info(set_color("=" * 72, "green"))
    logger.info(f"Best validation loss: {best_loss:.6f}")
    logger.info(f"Best collision rate: {best_collision_rate:.6f}")


if __name__ == "__main__":
    main()
