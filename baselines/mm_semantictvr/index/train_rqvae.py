import argparse
import torch
import numpy as np
from time import time
import logging
import os
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - training metrics won't be logged to wandb")

from torch.utils.data import DataLoader

from .datasets import VideoTextGuidedDataset, load_internvideo2_features
from .models.rqvae import RQVAE
from .trainer import Trainer
from .utils import seed_everything, setup_logging_with_file

def parse_args():
    parser = argparse.ArgumentParser(description="Text-Guided Video RQ-VAE Training")

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=1, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="msrvtt",
                        choices=['msrvtt', 'didemo', 'actnet', 'lsmdc', 'activitynet'],
                        help="Dataset name")
    parser.add_argument("--features_root", type=str,
                        default="./dataset/features",
                        help="Path to features directory")

    # Model hyperparameters
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="reconstruction loss type")
    parser.add_argument("--text_loss_type", type=str, default="mse", 
                        choices=['mse', 'l1', 'contrastive'], help="text guidance loss type")
    parser.add_argument("--text_loss_pos", type=str, default="after", 
                        choices=['before', 'after'], help="text loss position: before/after quantization")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=0.0, help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # RQ-VAE architecture (768D input from avg-pooled video)
    parser.add_argument('--code_num', type=int, default=256, help='number of codes per quantization layer')
    parser.add_argument('--codebook_layers', type=int, default=4, help='number of quantization layers in RQ-VAE')
    parser.add_argument('--e_dim', type=int, default=512, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512], help='hidden sizes of encoder/decoder layers')

    parser.add_argument("--ckpt_dir", type=str, default="index/log", help="output directory for model")
    
    # Wandb configuration
    parser.add_argument("--use_wandb", type=bool, default=True, help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="semantic-tvr", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")

    # Training mode
    parser.add_argument("--text_guided", action='store_true', help="use text guidance for training")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create checkpoint directory
    dataset_name = args.dataset
    mode_suffix = "text_guided" if args.text_guided else "standard"
    ckpt_subdir = f"{dataset_name}/{mode_suffix}/code_num_{args.code_num}_codebook_layers_{args.codebook_layers}"
    args.ckpt_dir = os.path.join(args.ckpt_dir, ckpt_subdir)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Setup logging to both console and file from the beginning
    train_log_path = os.path.join(args.ckpt_dir, "train.log")
    logger = setup_logging_with_file(train_log_path, level=logging.DEBUG)
    
    logger.info(f"Arguments: {args}")
    
    seed_everything(args.seed)

    # Pre-load feature dictionaries (saves time across worker processes)
    train_vid, train_txt, test_vid, test_txt = load_internvideo2_features(
        args.dataset, args.features_root
    )

    # Build datasets
    mode_str = "text-guided" if args.text_guided else "standard"
    print(f"Loading {dataset_name.upper()} dataset in {mode_str} mode...")
    train_data = VideoTextGuidedDataset(args.dataset, args.features_root, split="train", text_guided=args.text_guided, model_type='rqvae', feature_extractor='InternVideo2', video_features=train_vid, text_features=train_txt)
    valid_data = VideoTextGuidedDataset(args.dataset, args.features_root, split="train", text_guided=False, model_type='rqvae', feature_extractor='InternVideo2', video_features=train_vid, text_features=train_txt)
    test_data = VideoTextGuidedDataset(args.dataset, args.features_root, split="test", text_guided=False, model_type='rqvae', feature_extractor='InternVideo2', video_features=test_vid, text_features=test_txt)
    
    # Generate num_emb_list from code_num and codebook_layers
    num_emb_list = [args.code_num] * args.codebook_layers
    print(f"RQ-VAE configuration: {args.codebook_layers} layers with {args.code_num} codes each -> {num_emb_list}")
    
    # Initialize RQ-VAE model with video embedding dimensions (768D)
    model = RQVAE(in_dim=train_data.dim,
                  num_emb_list=num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  text_loss_type=args.text_loss_type,
                  text_loss_pos=args.text_loss_pos,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters)
    
    print(f"Model architecture:")
    print(model)
    print(f"Input dimension: {train_data.dim}")
    data_type = "video-text pairs" if args.text_guided else "video samples"
    print(f"Dataset: {dataset_name.upper()} with {len(train_data)} {data_type} in {mode_str} mode")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
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
        wandb_run_name = args.wandb_run_name or f"{dataset_name}_rqvae_{mode_str}_bs_{args.batch_size}_lr_{args.lr}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args)
        )
        print(f"Wandb initialized: {args.wandb_project}/{wandb_run_name}")
    
    # Initialize trainer and start training
    trainer = Trainer(args, model, video_key='video_emb', model_type='rqvae')
    best_loss, best_collision_rate = trainer.fit(train_data_loader, valid_data_loader, test_data_loader)
    
    # Close wandb if used
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print(f"Training completed!")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Best Collision Rate: {best_collision_rate:.6f}")


if __name__ == '__main__':
    main()
