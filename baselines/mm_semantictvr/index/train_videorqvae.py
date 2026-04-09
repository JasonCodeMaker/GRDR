import argparse
import torch
import numpy as np
from time import time
import logging
import os
import pickle
from datetime import datetime
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available - training metrics won't be logged to wandb")

from torch.utils.data import DataLoader

from .datasets import VideoTextGuidedDataset, MultiTextVideoDataset
from .models.rqvae import VideoRQVAE, VideoRQVAE_V2
from .trainer import Trainer
from .utils import seed_everything, set_color, setup_logging_with_file

def load_video_text_features(dataset_name, features_root, feature_extractor="InternVL", frames=8):
    """
    Load video and text features from pickle files for efficiency.

    Args:
        dataset_name: 'msrvtt', 'didemo', 'activitynet', etc.
        features_root: Root path to features directory
        feature_extractor: 'CLIP' or 'InternVL'
        frames: number of frames
    Returns:
        tuple: (train_video_features, train_text_features, test_video_features, test_text_features)
    """
    logger = logging.getLogger()
    logger.info(set_color(f"Loading {feature_extractor} features for {dataset_name.upper()}...", "green"))
    
    # Set features path based on feature_extractor
    if feature_extractor == "CLIP":
        features_path = os.path.join(features_root, "CLIP")
        feature_suffix = "cliplargel14"
    elif feature_extractor == "InternVL":
        features_path = os.path.join(features_root, "InternVL")
        feature_suffix = "internvl-hico-r16"
    elif feature_extractor == "InternVideo2":
        features_path = os.path.join(features_root, "InternVideo2")
        feature_suffix = "internvideo2"
    else:
        raise ValueError(f"Unsupported feature_extractor: {feature_extractor}. Must be 'CLIP' or 'InternVL'")
    
    # Convert to absolute path
    features_path = os.path.abspath(features_path)
    
    # Load train video features
    if feature_extractor == "InternVideo2":
        train_video_path = os.path.join(features_path, f"video/{dataset_name}_{feature_suffix}_video_embeddings_train.pkl")
    elif feature_extractor == "InternVL":
        train_video_path = os.path.join(features_path, f"{frames}/{dataset_name}_{feature_suffix}_video_embeddings_train.pkl")
        if not os.path.exists(train_video_path):
            raise FileNotFoundError(f"Train video features not found: {train_video_path}")
        
    logger.info(f"Loading train video features from: {train_video_path}")
    start_time = time()
    with open(train_video_path, 'rb') as f:
        train_video_features = pickle.load(f)
    logger.info(f"Loaded {len(train_video_features)} train video features in {time() - start_time:.2f} seconds")
    
    # Load train text features
    train_text_path = os.path.join(features_path, f"{dataset_name}_{feature_suffix}_text_embeddings_train.pkl")
    train_text_features = None
    if os.path.exists(train_text_path):
        logger.info(f"Loading train text features from: {train_text_path}")
        start_time = time()
        with open(train_text_path, 'rb') as f:
            train_text_features = pickle.load(f)
        logger.info(f"Loaded {len(train_text_features)} train text features in {time() - start_time:.2f} seconds")
    else:
        logger.warning(f"Train text features not found: {train_text_path}")
    
    # Load test video features
    if feature_extractor == "InternVideo2":
        test_video_path = os.path.join(features_path, f"video/{dataset_name}_{feature_suffix}_video_embeddings_test.pkl")
    elif feature_extractor == "InternVL":
        test_video_path = os.path.join(features_path, f"{frames}/{dataset_name}_{feature_suffix}_video_embeddings_test.pkl")
    test_video_features = None
    if os.path.exists(test_video_path):
        logger.info(f"Loading test video features from: {test_video_path}")
        start_time = time()
        with open(test_video_path, 'rb') as f:
            test_video_features = pickle.load(f)
        logger.info(f"Loaded {len(test_video_features)} test video features in {time() - start_time:.2f} seconds")
    else:
        logger.warning(f"Test video features not found: {test_video_path}")

    # Load test text features for text reconstruction loss evaluation
    test_text_path = os.path.join(features_path, f"{dataset_name}_{feature_suffix}_text_embeddings_test.pkl")
    test_text_features = None
    if os.path.exists(test_text_path):
        logger.info(f"Loading test text features from: {test_text_path}")
        start_time = time()
        with open(test_text_path, 'rb') as f:
            test_text_features = pickle.load(f)
        logger.info(f"Loaded {len(test_text_features)} test text features in {time() - start_time:.2f} seconds")
    else:
        logger.warning(f"Test text features not found: {test_text_path}")

    return train_video_features, train_text_features, test_video_features, test_text_features

def parse_args():
    parser = argparse.ArgumentParser(description="Text-Guided Video RQ-VAE Training")

    # Experiment parameters
    parser.add_argument('--version', type=str, default="4.0", help="version of the model")
    parser.add_argument('--device', type=int, default=1, help="which CUDA device to use")
    parser.add_argument("--ckpt_dir", type=str, default="./index/log", help="output directory for model")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=5, help='eval step')

    # Learning rate parameters
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--vq_lr', type=float, default=1e-2, help='learning rate for VectorQuantizer parameters')
    parser.add_argument('--lr_scheduler', type=str, default="cosine",
                        choices=['cosine', 'linear', 'constant'],
                        help='learning rate scheduler type')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='warmup ratio for learning rate scheduler')
    parser.add_argument('--min_lr_ratio', type=float, default=0.0,
                        help='minimum learning rate as ratio of initial learning rate')
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="msrvtt", 
                        choices=['msrvtt', 'didemo', 'actnet', 'activitynet'],
                        help="Dataset name")
    parser.add_argument("--features_root", type=str,
                        default="./data_process/datasets/features",
                        help="Path to features directory")
    parser.add_argument("--feature_extractor", type=str, default="InternVideo2",
                        choices=['CLIP', 'InternVL', 'InternVideo2'], help="feature extractor type")
    parser.add_argument("--frames", type=int, default=8, help="number of frames")

    # Model hyperparameters
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='l2 regularization weight (increased to reduce overfitting)')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", action="store_true", default=False, help="use batch normalization")
    parser.add_argument("--vid_loss_weight", type=float, nargs=5, default=[1.0, 0.0, 1.0, 0.0, 0.0],
                        help="Video reconstruction loss weights [mse, l1, cosine, cls, p2p_infonce]. Use 0 to disable a loss.")
    parser.add_argument("--text_loss_type", type=str, default="contrastive",
                        choices=['mse', 'contrastive'], help="text guidance loss type")
    parser.add_argument("--text_loss_pos", type=str, default="after",
                        choices=['before', 'after'], help="text loss position: before/after quantization")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07,
                        help="temperature for InfoNCE contrastive loss")
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0,
                        help="weight for text-video contrastive loss (0 to disable)")
    parser.add_argument("--no_kmeans_init", dest="kmeans_init", action="store_false", default=True, help="disable kmeans initialization")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=0.0, help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--beta", type=float, default=0.35, help="beta for vq loss (commitment cost - increased to reduce overfitting)")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="use EMA (Exponential Moving Average) for codebook updates instead of gradients (enabled by default to reduce overfitting)")
    parser.add_argument("--ema_decay", type=float, default=0.99,
                        help="decay rate for EMA updates (only used if use_ema=True, increased to 0.99 for better stability)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # VideoRQ-VAE architecture (patch-level processing)
    parser.add_argument('--code_num', type=int, default=256, help='number of codes per quantization layer')
    parser.add_argument('--codebook_layers', type=int, default=4, help='number of quantization layers in RQ-VAE')
    parser.add_argument('--e_dim', type=int, default=512, help='vq codebook embedding size (latent dimension)')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')

    # VideoRQVAE specific architecture parameters
    parser.add_argument('--encoder_width', type=int, default=512, help='encoder transformer width')
    parser.add_argument('--encoder_layers', type=int, default=2, help='number of encoder transformer layers')
    parser.add_argument('--encoder_heads', type=int, default=8, help='number of encoder attention heads')
    parser.add_argument('--num_latent_tokens', type=int, default=4, help='number of latent tokens for quantization')
    parser.add_argument('--encoder', type=str, default="VideoLatentEncoder", help='encoder type')
    parser.add_argument('--decoder', type=str, default="VideoDecoder", help='decoder type')

    # Wandb configuration
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false", default=True, help="disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="semantic-tvr", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")

    # Text-guided training mode
    parser.add_argument("--text_mode", type=str, default="reconstruction",
                        choices=['none', 'guided', 'reconstruction'],
                        help="Text training mode: 'none' (video-only), 'guided' (text guidance loss), 'reconstruction' (text decoder + guidance)")
    parser.add_argument("--text_dim", type=int, default=4096, help="target text embedding dimension")
    parser.add_argument("--text_decoder_layers", type=int, nargs='+', default=[1536, 3072], help="hidden layers for text decoder")
    parser.add_argument("--multi_text_mode", action="store_true", help="use multi-text training with all captions per video")

    # Router parameters
    parser.add_argument("--router_hidden_dim", type=int, default=512, help="hidden dimension for router")
    parser.add_argument("--router_temperature", type=float, default=1.0, help="temperature for router softmax")

    # Text reconstruction loss parameter
    parser.add_argument("--text_recon_loss_weight", type=float, default=0.5, help="weight for text reconstruction loss")

    # Router diversity loss weight (only applied when num_latent_tokens > 1)
    parser.add_argument("--diversity_loss_weight", type=float, default=10.0, help="weight for router diversity loss")

    return parser.parse_args()


def main():
    args = parse_args()

    # Derive text configuration from text_mode (needed for checkpoint path)
    args.text_guided = args.text_mode in ['guided', 'reconstruction']
    args.use_text_decoder = args.text_mode == 'reconstruction'

    # VideoRQVAE_V2 specific configuration adjustments for InternVideo2
    if args.feature_extractor == "InternVideo2":
        # VideoRQVAE_V2 works with pooled features (single vector per video)

        # Enable multi_text_mode automatically for contrastive learning
        if args.contrastive_loss_weight > 0 and not args.multi_text_mode:
            print(f"INFO: Enabling multi_text_mode for contrastive learning with InternVideo2")
            args.multi_text_mode = True

        # Text decoder is not yet supported for V2 architecture
        if args.use_text_decoder:
            print(f"WARNING: text_decoder is not yet supported for InternVideo2 (VideoRQVAE_V2). Disabling text_decoder.")
            args.use_text_decoder = False
            # But keep text_guided=True if using contrastive learning
            if args.contrastive_loss_weight == 0:
                args.text_guided = False
            args.text_mode = 'none'

    # Create checkpoint directory
    dataset_name = args.dataset
    mode_suffix = args.text_mode
    ckpt_subdir = f"{dataset_name}_{mode_suffix}_v{args.version}/lr_{args.lr}_vr_{args.vq_lr}_n_{args.num_latent_tokens}_cn_{args.code_num}_cl_{args.codebook_layers}_beta_{args.beta}_dl_{args.diversity_loss_weight}"
    date_suffix = datetime.now().strftime("%m%d_%H")
    args.ckpt_dir = os.path.join(args.ckpt_dir, ckpt_subdir, date_suffix)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Setup logging to both console and file from the beginning
    train_log_path = os.path.join(args.ckpt_dir, "train.log")
    logger = setup_logging_with_file(train_log_path, level=logging.INFO)
    
    logger.info(set_color("=" * 80, "blue"))
    logger.info(set_color("VideoRQVAE Training Configuration", "blue"))
    logger.info(set_color("=" * 80, "blue"))

    # Set CUDA device with basic validation
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a system with at least one GPU.")
    torch.cuda.set_device(args.device)
    logger.info(f"Using CUDA device: {args.device}")

    seed_everything(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Load video and text features once for efficiency
    logger.info(set_color("Pre-loading video and text features...", "green"))
    train_video_features, train_text_features, test_video_features, test_text_features = load_video_text_features(
        dataset_name, args.features_root, feature_extractor=args.feature_extractor, frames=args.frames
    )
    
    # Build datasets based on training mode
    if args.multi_text_mode:
        logger.info(set_color(f"Creating {dataset_name.upper()} MULTI-TEXT dataset instances for VideoRQVAE...", "green"))
        train_data = MultiTextVideoDataset(
            args.dataset, args.features_root, split="train",
            feature_extractor=args.feature_extractor,
            video_features=train_video_features, text_features=train_text_features
        )
        # For validation, use multi-text mode
        valid_data = MultiTextVideoDataset(
            args.dataset, args.features_root, split="train",
            feature_extractor=args.feature_extractor,
            video_features=train_video_features, text_features=train_text_features
        )
        # Test data with single GT text queries for text reconstruction loss evaluation
        test_data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split="test", text_guided=True,
            feature_extractor=args.feature_extractor, model_type="videorqvae",
            video_features=test_video_features, text_features=test_text_features
        )
    else:
        logger.info(set_color(f"Creating {dataset_name.upper()} dataset instances in {args.text_mode} mode for VideoRQVAE...", "green"))
        train_data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split="train", text_guided=args.text_guided,
            feature_extractor=args.feature_extractor, model_type="videorqvae",
            video_features=train_video_features, text_features=train_text_features
        )
        valid_data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split="train", text_guided=args.text_guided,
            feature_extractor=args.feature_extractor, model_type="videorqvae",
            video_features=train_video_features, text_features=train_text_features
        )
        # Test data with single GT text queries for text reconstruction loss evaluation
        test_data = VideoTextGuidedDataset(
            args.dataset, args.features_root, split="test", text_guided=True,
            feature_extractor=args.feature_extractor, model_type="videorqvae",
            video_features=test_video_features, text_features=test_text_features
        )
    
    # Generate num_emb_list from code_num and codebook_layers
    num_emb_list = [args.code_num] * args.codebook_layers
    logger.info(set_color(f"VideoRQVAE configuration: {args.codebook_layers} layers with {args.code_num} codes each -> {num_emb_list}", "blue"))

    # Get dataset dimensions
    if hasattr(train_data, 'video_text_groups') and train_data.video_ids:
        # MultiTextVideoDataset
        sample_video = train_data.video_text_groups[train_data.video_ids[0]]['video']
        if hasattr(sample_video, 'shape'):
            if len(sample_video.shape) == 1:
                num_patches, dim = 1, sample_video.shape[0]
            else:
                num_patches, dim = sample_video.shape
        else:
            num_patches, dim = train_data.num_patches, train_data.dim
    else:
        # VideoTextGuidedDataset
        num_patches, dim = train_data.num_patches, train_data.dim

    # Construct model configuration from args (single source of truth)
    model_config = {
        'in_dim': dim,
        'num_patches': num_patches,
        'encoder_width': args.encoder_width,
        'encoder_layers': args.encoder_layers,
        'encoder_heads': args.encoder_heads,
        'num_latent_tokens': args.num_latent_tokens,
        'num_emb_list': num_emb_list,
        'e_dim': args.e_dim,
        'dropout_prob': args.dropout_prob,
        'bn': args.bn,
        'vid_loss_weight': args.vid_loss_weight,
        'text_loss_type': args.text_loss_type,
        'text_loss_pos': args.text_loss_pos,
        'quant_loss_weight': args.quant_loss_weight,
        'kmeans_init': args.kmeans_init,
        'kmeans_iters': args.kmeans_iters,
        'sk_epsilons': args.sk_epsilons,
        'sk_iters': args.sk_iters,
        'beta': args.beta,
        'use_ema': args.use_ema,
        'ema_decay': args.ema_decay,
        'use_text_decoder': args.use_text_decoder,
        'text_dim': args.text_dim,
        'text_decoder_layers': args.text_decoder_layers,
        'router_hidden_dim': args.router_hidden_dim,
        'router_temperature': args.router_temperature,
        'text_recon_loss_weight': args.text_recon_loss_weight,
        'diversity_loss_weight': args.diversity_loss_weight,
        'contrastive_temperature': args.contrastive_temperature,
        'contrastive_loss_weight': args.contrastive_loss_weight,
        'num_frames': args.frames,
        'frame_temperature': 0.1,
        'version': args.version,
        'encoder': args.encoder,
        'decoder': args.decoder,
    }

    # Initialize VideoRQVAE model from config
    if args.feature_extractor == "InternVideo2":
        model = VideoRQVAE_V2(**model_config)
    else:
        model = VideoRQVAE(**model_config)

    # Add metadata to model config after initialization (for checkpoint saving)
    model.config['feature_extractor'] = args.feature_extractor

    logger.info(set_color("=" * 80, "blue"))
    logger.info(set_color("Model Architecture", "blue"))
    logger.info(set_color("=" * 80, "blue"))
    logger.info(f"Input patches: {num_patches}")
    logger.info(f"Feature dimension per patch: {dim}")
    logger.info(f"Feature extractor: {args.feature_extractor}")

    if args.multi_text_mode:
        logger.info(f"Dataset: {dataset_name.upper()} with {len(train_data)} videos in MULTI-TEXT mode")
        if hasattr(train_data, 'video_text_groups') and train_data.video_ids:
            sample_texts = train_data.video_text_groups[train_data.video_ids[0]]['texts']
            logger.info(f"Texts per video: {len(sample_texts)}")
    else:
        data_type = "video-text pairs" if args.text_guided else "video patch sequences"
        logger.info(f"Dataset: {dataset_name.upper()} with {len(train_data)} {data_type} in {args.text_mode} mode")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"VideoRQVAE configured for patch-level processing with {num_patches} patches")

    # Display text configuration based on mode
    logger.info(set_color("=" * 80, "blue"))
    logger.info(set_color("Text Configuration", "blue"))
    logger.info(set_color("=" * 80, "blue"))
    if args.multi_text_mode:
        logger.info(f"Multi-text mode: Training with all captions per video simultaneously")
        logger.info(f"Text decoder: text_dim={args.text_dim}, layers={args.text_decoder_layers}")
        logger.info(f"Router config: hidden_dim={args.router_hidden_dim}, temperature={args.router_temperature}")
        logger.info(f"Text reconstruction loss weight: {args.text_recon_loss_weight}")
        if args.num_latent_tokens > 1:
            logger.info(f"Router diversity loss weight: {args.diversity_loss_weight} (ENABLED)")
        else:
            logger.info(f"Router diversity loss: DISABLED (num_latent_tokens = 1)")
    elif args.text_mode == 'none':
        logger.info(f"Text mode: Video-only training (no text components)")
    elif args.text_mode == 'guided':
        logger.info(f"Text mode: Text-guided training (text guidance loss only)")
        logger.info(f"Text loss: {args.text_loss_type} at {args.text_loss_pos} quantization")
    elif args.text_mode == 'reconstruction':
        logger.info(f"Text mode: Full text reconstruction training")
        logger.info(f"Text guidance loss: {args.text_loss_type} at {args.text_loss_pos} quantization")
        logger.info(f"Text decoder: text_dim={args.text_dim}, layers={args.text_decoder_layers}")
        logger.info(f"Router config: hidden_dim={args.router_hidden_dim}, temperature={args.router_temperature}")
        logger.info(f"Text reconstruction loss weight: {args.text_recon_loss_weight}")

    # Display loss configuration
    logger.info(set_color("=" * 80, "blue"))
    logger.info(set_color("Loss Configuration", "blue"))
    logger.info(set_color("=" * 80, "blue"))

    # Video Reconstruction Losses
    logger.info(set_color("Video Reconstruction Losses:", "yellow"))
    vid_loss_names = ['MSE', 'L1', 'Cosine', 'Frame Classification', 'Patch-to-Patch InfoNCE']
    active_vid_losses = []
    for idx, (name, weight) in enumerate(zip(vid_loss_names, args.vid_loss_weight)):
        if weight > 0:
            logger.info(f"  {name}: weight = {weight}")
            active_vid_losses.append(name)
        else:
            logger.info(f"  {name}: DISABLED (weight = 0)")

    # Quantization Loss
    logger.info(set_color("Quantization Loss:", "yellow"))
    logger.info(f"  RQ-VAE quantization loss weight: {args.quant_loss_weight}")
    logger.info(f"  Beta (commitment cost): {args.beta}")
    
    # EMA Configuration
    logger.info(set_color("Codebook Update Strategy:", "yellow"))
    if args.use_ema:
        logger.info(f"  Update method: EMA (Exponential Moving Average)")
        logger.info(f"  EMA decay: {args.ema_decay}")
        logger.info(f"  Note: Codebook updated via EMA statistics (no gradients)")
        logger.info(f"  Note: Sinkhorn disabled when using EMA")
    else:
        logger.info(f"  Update method: Gradient-based")
        logger.info(f"  Loss components: Codebook loss + {args.beta} * Commitment loss")

    # Text-related Losses
    if args.text_mode != 'none':
        logger.info(set_color("Text-related Losses:", "yellow"))

        # Text Guidance Loss (applied during forward pass for text-guided training)
        if args.text_guided:
            logger.info(f"  Text guidance loss: {args.text_loss_type} (applied {args.text_loss_pos} quantization)")
            if args.text_loss_type == 'contrastive':
                logger.info(f"    Temperature: {args.contrastive_temperature}")

        # Text Reconstruction Loss (when text decoder is enabled)
        if args.use_text_decoder:
            logger.info(f"  Text reconstruction loss: {args.text_loss_type}")
            logger.info(f"  Weight: {args.text_recon_loss_weight}")
            if args.text_loss_type == 'contrastive':
                logger.info(f"    Temperature: {args.contrastive_temperature}")

        # Router Diversity Loss
        if args.num_latent_tokens > 1 and args.diversity_loss_weight > 0:
            logger.info(f"  Router diversity loss: weight = {args.diversity_loss_weight}")
        elif args.num_latent_tokens > 1:
            logger.info(f"  Router diversity loss: DISABLED (weight = 0)")

    # Additional loss-related parameters
    logger.info(set_color("Additional Loss Parameters:", "yellow"))
    if args.vid_loss_weight[4] > 0:
        logger.info(f"  Patch-to-patch InfoNCE temperature: {args.contrastive_temperature}")

    # Create train_data loader
    train_data_loader = DataLoader(train_data, 
                           num_workers=args.num_workers,
                           batch_size=args.batch_size, 
                           shuffle=True,
                           pin_memory=True)
    
    valid_data_loader = DataLoader(valid_data, 
                           num_workers=args.num_workers,
                           batch_size=args.batch_size, 
                           shuffle=False,
                           pin_memory=True)

    test_data_loader = DataLoader(test_data, 
                           num_workers=args.num_workers,
                           batch_size=args.batch_size, 
                           shuffle=False,
                           pin_memory=True)
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_run_name = args.wandb_run_name or f"v{args.version}_n_{args.num_latent_tokens}_cn_{args.code_num}_cl_{args.codebook_layers}_lr_{args.lr}_vr_{args.vq_lr}_beta_{args.beta}_dl_{args.diversity_loss_weight}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args)
        )
        logger.info(set_color(f"Wandb initialized: {args.wandb_project}/{wandb_run_name}", "green"))
    
    # Initialize trainer and start training
    video_key = 'video_patches' if (args.text_guided or args.multi_text_mode) else None
    trainer = Trainer(args, model, video_key, 'videorqvae')

    logger.info(set_color("=" * 80, "blue"))
    logger.info(set_color("Starting Training", "blue"))
    logger.info(set_color("=" * 80, "blue"))
    
    best_loss, best_collision_rate = trainer.fit(train_data_loader, valid_data_loader, test_data_loader)
    
    # Close wandb if used
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    logger.info(set_color("=" * 80, "green"))
    logger.info(set_color("Training Completed!", "green"))
    logger.info(set_color("=" * 80, "green"))
    logger.info(f"Dataset: {dataset_name.upper()}")
    logger.info(f"Best Loss: {best_loss:.6f}")
    logger.info(f"Best Collision Rate: {best_collision_rate:.6f}")


if __name__ == '__main__':
    main()
