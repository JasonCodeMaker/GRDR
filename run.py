import copy
import json
import os
import argparse
import time

import torch
import wandb
from tqdm import tqdm

from models.grdr import GRDR, Codebook, QuantizeOutput, VideoOutput
from trainer.trainer import OurTrainer, train, build_loss_weights
from trainer.evaluator import test, test_dr, eval_retrieval
from utils.model_utils import seed_everything


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='t5-small',
                        choices=['google/t5-efficient-tiny', 't5-small', 't5-base', 't5-large', 't5-3b'],
                        help='HuggingFace model name')
    parser.add_argument('--code_num', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size used by DataLoader')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (fallback)')

    # Phase-specific learning rates
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='Learning rate for pre-train phase')
    parser.add_argument('--main_lr', type=float, default=1e-4, help='Learning rate for main training phase')
    parser.add_argument('--fit_lr', type=float, default=1e-4, help='Learning rate for fit phase')

    # Phase-specific epochs
    parser.add_argument('--pretrain_epochs', type=int, default=1, help='Number of epochs for pre-train phase')
    parser.add_argument('--main_epochs', type=int, default=2, help='Number of epochs for main training phase')
    parser.add_argument('--fit_epochs', type=int, default=2, help='Number of epochs for fit phase')

    # Loss weights - Phase 1 (Pre-train): cl_loss + code_loss
    parser.add_argument('--w1_cl_loss', type=float, default=0.5, help='Phase 1 contrastive loss weight')
    parser.add_argument('--w1_ce_loss', type=float, default=0, help='Phase 1 cross-entropy loss weight')
    parser.add_argument('--w1_code_loss', type=float, default=0.5, help='Phase 1 code prediction loss weight')
    parser.add_argument('--w1_cl_dd_loss', type=float, default=0, help='Phase 1 video reconstruction loss weight')
    parser.add_argument('--w1_rq_loss', type=float, default=0, help='Phase 1 RQ quantization loss weight')

    # Loss weights - Phase 2 (Main Training): ce_loss + code_loss + cl_dd_loss + rq_loss
    parser.add_argument('--w2_cl_loss', type=float, default=0.2, help='Phase 2 contrastive loss weight')
    parser.add_argument('--w2_ce_loss', type=float, default=0.5, help='Phase 2 cross-entropy loss weight')
    parser.add_argument('--w2_code_loss', type=float, default=0.8, help='Phase 2 code prediction loss weight')
    parser.add_argument('--w2_cl_dd_loss', type=float, default=0.1, help='Phase 2 video reconstruction loss weight')
    parser.add_argument('--w2_rq_loss', type=float, default=0.3, help='Phase 2 RQ quantization loss weight')

    # Loss weights - Phase 2 (Optional): Fit phase
    parser.add_argument('--enable_fit', action='store_true', default=True, help='Enable fit phase')
    parser.add_argument('--w3_cl_loss', type=float, default=0, help='Phase fit contrastive loss weight')
    parser.add_argument('--w3_ce_loss', type=float, default=1, help='Phase fit cross-entropy loss weight')
    parser.add_argument('--w3_code_loss', type=float, default=1, help='Phase fit code prediction loss weight')
    parser.add_argument('--w3_cl_dd_loss', type=float, default=0, help='Phase fit video reconstruction loss weight')
    parser.add_argument('--w3_rq_loss', type=float, default=0, help='Phase fit RQ quantization loss weight')

    # Dataset arguments (for video-text integration)
    parser.add_argument('--dataset', type=str, default='msrvtt',
                       choices=['msrvtt', 'actnet', 'didemo', 'lsmdc'],
                       help='Dataset name for video-text features')
    parser.add_argument('--features_root', type=str, default='dataset/features',
                       help='Root directory for InternVideo2 features')
    parser.add_argument('--videorqvae_checkpoint', type=str,
                       default=None,
                       help='VideoRQVAE checkpoint path for code generation (optional, creates new model if not provided)')
    parser.add_argument('--num_latent_tokens', type=int, default=4,
                       help='Number of latent tokens in VideoRQVAE')

    # Evaluation arguments
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the model')
    parser.add_argument('--eval_checkpoint', type=str,
                        default="output/GRDR/msrvtt/current_best/model-3-fit/best_model.pt",
                        help='Checkpoint path for evaluation')
    parser.add_argument('--num_candidates', type=int, default=10,
                       help='Number of top candidates to retrieve per query for JSON export')
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2],
                       help='Setting: 1=test only pool, 2=train+test combined pool')
    parser.add_argument('--detailed_generation', action='store_true', default=False,
                       help='Include (sID, video_id) pairs in candidates and ground_truth_sID in output')

    parser.add_argument('--save_path', type=str, default='output/GRDR')
    parser.add_argument('--exp_name', type=str, default='debug', help='Experiment name for wandb and save path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=int, default=0, choices=[0, 1],
                       help='GPU device ID to use for training (0 or 1)')
    parser.add_argument('--use_pseudo_queries', action='store_true', default=False,
                       help='Include pseudo queries in training data')

    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


def main():
    """Main entry point for training and evaluation."""
    args = parse_args()

    # Set CUDA device before any CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    print(f'Using GPU: {args.device} (CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]})')

    seed_everything(args.seed)
    config = copy.deepcopy(vars(args))

    if args.eval:
        config['eval_checkpoint'] = args.eval_checkpoint
        test(config)
    else:
        timestamp = time.strftime('%Y%m%d%H%M%S')
        exp_segment = f'{timestamp}-{args.exp_name}' if args.exp_name else timestamp
        save_root = os.path.join(args.save_path, f'{args.dataset}/{exp_segment}')

        # Initialize wandb
        project_name = f"{config['dataset']}_GRDR"
        wandb.init(project=project_name, name=args.exp_name or None, config=config)

        checkpoint = None
        global_step = 0

        for loop in range(args.max_length):
            # Phase 1: Pre-train
            config['loop'] = loop
            config['save_path'] = os.path.join(save_root, f'model-{loop + 1}-pre')
            config['code_length'] = loop + 1
            config['prev_model'] = checkpoint
            config['prev_id'] = f'{checkpoint}.code' if checkpoint is not None else None
            config['epochs'] = 3 if loop == 0 else args.pretrain_epochs
            config['loss_w'] = 1
            config['lr'] = args.pretrain_lr
            checkpoint, global_step = train(config, global_step)
            test_dr(config, checkpoint)

            # Phase 2: Main Training
            config['save_path'] = os.path.join(save_root, f'model-{loop + 1}')
            config['prev_model'] = checkpoint
            config['codebook_init'] = f'{checkpoint}.kmeans.{args.code_num}'
            config['epochs'] = args.main_epochs
            config['loss_w'] = 2
            config['lr'] = args.main_lr
            checkpoint, global_step = train(config, global_step)
            if args.enable_fit:
                config['save_path'] = os.path.join(save_root, f'model-{loop+1}-fit')
                config['prev_model'] = checkpoint
                config['codebook_init'] = None
                config['epochs'] = args.fit_epochs
                config['loss_w'] = 3
                config['lr'] = args.fit_lr
                checkpoint, global_step = train(config, global_step)

        wandb.finish()


if __name__ == '__main__':
    main()
