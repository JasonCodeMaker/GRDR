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
from trainer.evaluator import test, test_dr
from utils.data_utils import has_kmeans_cache, load_shared_features
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
    parser.add_argument('--w1_route_agree_loss', type=float, default=0,
                        help='Phase 1 semantic-ID route agreement loss weight')
    parser.add_argument('--w1_bucket_route_loss', type=float, default=0,
                        help='Phase 1 bucket-aware semantic-ID route loss weight')
    parser.add_argument('--w1_video_rank_loss', type=float, default=0,
                        help='Phase 1 bucket-penalized video-route ranking loss weight')
    parser.add_argument('--w1_expanded_size_loss', type=float, default=0,
                        help='Phase 1 expected expanded bucket-size regularization weight')

    # Loss weights - Phase 2 (Main Training): ce_loss + code_loss + cl_dd_loss + rq_loss
    parser.add_argument('--w2_cl_loss', type=float, default=0.2, help='Phase 2 contrastive loss weight')
    parser.add_argument('--w2_ce_loss', type=float, default=0.5, help='Phase 2 cross-entropy loss weight')
    parser.add_argument('--w2_code_loss', type=float, default=0.8, help='Phase 2 code prediction loss weight')
    parser.add_argument('--w2_cl_dd_loss', type=float, default=0.1, help='Phase 2 video reconstruction loss weight')
    parser.add_argument('--w2_rq_loss', type=float, default=0.3, help='Phase 2 RQ quantization loss weight')
    parser.add_argument('--w2_route_agree_loss', type=float, default=0,
                        help='Phase 2 semantic-ID route agreement loss weight')
    parser.add_argument('--w2_bucket_route_loss', type=float, default=0,
                        help='Phase 2 bucket-aware semantic-ID route loss weight')
    parser.add_argument('--w2_video_rank_loss', type=float, default=0,
                        help='Phase 2 bucket-penalized video-route ranking loss weight')
    parser.add_argument('--w2_expanded_size_loss', type=float, default=0,
                        help='Phase 2 expected expanded bucket-size regularization weight')

    # Loss weights - Phase 2 (Optional): Fit phase
    parser.add_argument('--enable_fit', action=argparse.BooleanOptionalAction, default=True, help='Enable fit phase')
    parser.add_argument('--w3_cl_loss', type=float, default=0, help='Phase fit contrastive loss weight')
    parser.add_argument('--w3_ce_loss', type=float, default=1, help='Phase fit cross-entropy loss weight')
    parser.add_argument('--w3_code_loss', type=float, default=1, help='Phase fit code prediction loss weight')
    parser.add_argument('--w3_cl_dd_loss', type=float, default=0, help='Phase fit video reconstruction loss weight')
    parser.add_argument('--w3_rq_loss', type=float, default=0, help='Phase fit RQ quantization loss weight')
    parser.add_argument('--w3_route_agree_loss', type=float, default=0,
                        help='Phase fit semantic-ID route agreement loss weight')
    parser.add_argument('--w3_bucket_route_loss', type=float, default=0.10,
                        help='Phase fit bucket-aware semantic-ID route loss weight')
    parser.add_argument('--w3_video_rank_loss', type=float, default=0,
                        help='Phase fit bucket-penalized video-route ranking loss weight')
    parser.add_argument('--w3_expanded_size_loss', type=float, default=0,
                        help='Phase fit expected expanded bucket-size regularization weight')
    parser.add_argument('--route_agree_stopgrad_video', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Detach video-side logits for route-aware losses')
    parser.add_argument('--route_bucket_gamma', type=float, default=1.0,
                        help='Inverse-bucket weighting exponent for bucket-aware route loss')
    parser.add_argument('--video_rank_beta', type=float, default=0.5,
                        help='Bucket-size penalty coefficient for video-rank loss')
    parser.add_argument('--route_bucket_default_size', type=float, default=1.0,
                        help='Fallback bucket size when a sample route is missing from saved bucket stats')

    # Dataset arguments (for video-text integration)
    parser.add_argument('--dataset', type=str, default='msrvtt',
                       choices=['msrvtt', 'actnet', 'didemo', 'lsmdc', 'panda'],
                       help='Dataset name for video-text features')
    parser.add_argument('--features_root', type=str, default='dataset/features',
                       help='Root directory for InternVideo2 features')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Directory for dataset-side caches such as text k-means groups')
    parser.add_argument('--videorqvae_checkpoint', type=str,
                       default=None,
                       help='VideoRQVAE checkpoint path for code generation (optional, creates new model if not provided)')
    parser.add_argument('--num_latent_tokens', type=int, default=4,
                       help='Number of latent tokens in VideoRQVAE')

    # Multi-view fix pipeline (research_html/packages/2026-05-24-multiview-fix-pipeline)
    # K1 path is now the default; ablate via --no-codebook_seed_all_slots.
    parser.add_argument('--codebook_seed_all_slots', action=argparse.BooleanOptionalAction, default=True,
                       help='K1: at every loop boundary, run faiss-GPU k-means on '
                            '[num_videos x N, D] features (dedup + return_all). Default on.')

    # Evaluation arguments
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the model')
    parser.add_argument('--eval_checkpoint', type=str,
                        default="output/GRDR/bucket_candidate_k20/msrvtt/20260428163014-fit_bucket_l010_g10_k20_s42/model-3-fit/best_model.pt",
                        help='Checkpoint path for evaluation')
    parser.add_argument('--num_candidates', type=int, default=20,
                       help='Number of constrained-generation beams/routes per query before sID expansion')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                       help='Batch size for retrieval evaluation; defaults to training batch size')
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2],
                       help='Setting: 1=test only pool, 2=train+test combined pool')
    parser.add_argument('--distractor_n', type=int, default=0,
                       help='Setting 2 only: if >0, deterministically subsample N train videos (seed fixed=42) before merging with test; 0 = use full train pool')
    parser.add_argument('--detailed_generation', action='store_true', default=False,
                       help='Include (sID, video_id) pairs in candidates and ground_truth_sID in output')
    # BARS reorder is the canonical evaluation strategy (research_html/packages/
    # 2026-05-15-panda-baselines + 2026-05-16-panda-pseudo-queries-multiview).
    # Default ON for every eval; disable via --no-inference_reorder_by_access_score.
    parser.add_argument('--inference_reorder_by_access_score', action=argparse.BooleanOptionalAction, default=True,
                       help='Reorder expanded candidates with BARS beam score plus bucket penalty. Default on.')
    parser.add_argument('--access_score_bucket_gamma', type=float, default=0.50,
                       help='Bucket-size penalty weight for BARS reorder. Canonical default 0.50.')
    parser.add_argument('--candidate_handoff_cap', type=int, default=0,
                       help='If >0, cap the final candidate JSON to this many videos per query after optional BARS reorder')
    parser.add_argument('--candidate_output_dir', type=str, default='candidates',
                       help='Directory for exported candidate JSON')
    parser.add_argument('--candidate_sidecar_dir', type=str, default=None,
                       help='Optional directory for per-query candidate audit JSONL sidecars')
    # Pass-B (efficiency) latency contract --- see
    # research_html/packages/2026-05-15-panda-baselines/docs/eval-efficiency.html
    parser.add_argument('--candidate_export', action='store_true', default=False,
                       help='Pass-B Stage-1 entry: run candidate export only (equivalent to --eval with latency-mode setup)')
    parser.add_argument('--subset_manifest', type=str, default=None,
                       help='Pass-B latency manifest; restricts queries to listed video_ids')
    parser.add_argument('--warmup_n_used', type=int, default=10,
                       help='Manifest warmup ids consumed before timing starts')
    parser.add_argument('--wall_time_cap_s', type=float, default=300.0,
                       help='Per-cell wall-time cap; stops between queries when exceeded')
    parser.add_argument('--latency_helpers_dir', type=str,
                       default='/home/uqzzha35/Project/SemanticID/GRDR/research_html/packages/2026-05-15-panda-baselines/scripts',
                       help='Directory containing latency_helpers.py')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Pass-B explicit output path for the candidate JSON (overrides --candidate_output_dir)')

    parser.add_argument('--save_path', type=str, default='output/GRDR/bucket_candidate_k20')
    parser.add_argument('--exp_name', type=str, default='fit_bucket_l010_g10_k20_s42', help='Experiment name for wandb and save path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=int, default=0, choices=[0, 1],
                       help='GPU device ID to use for training (0 or 1)')
    parser.add_argument('--use_pseudo_queries', action='store_true', default=False,
                       help='Include pseudo queries in training data')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Override wandb project name (default: f"{dataset}_GRDR" for back-compat).')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Override wandb run name (default: args.exp_name).')
    parser.add_argument('--start_loop', type=int, default=0,
                       help='Resume progressive training from this loop index (0-based). Use with --init_checkpoint to extend an existing l=start_loop ckpt with the next code layer.')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                       help='Initial checkpoint when --start_loop > 0; must be the model-{start_loop}-fit/best_model.pt of a prior run with the same codebook width.')
    parser.add_argument('--skip_pretrain', action='store_true', default=False,
                       help='When --init_checkpoint is a model-{start_loop+1}-pre/best_model.pt, skip that loop-{start_loop} pretrain train() call and reuse the init_checkpoint. Resumes a chain whose pretrain finished but whose subsequent kmeans/main/fit phases were interrupted.')

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

    if args.eval or args.candidate_export:
        config['eval_checkpoint'] = args.eval_checkpoint
        test(config)
    else:
        timestamp = time.strftime('%Y%m%d%H%M%S')
        exp_segment = f'{timestamp}-{args.exp_name}' if args.exp_name else timestamp
        save_root = os.path.join(args.save_path, f'{args.dataset}/{exp_segment}')

        # Initialize wandb (R-LOG: launchers can pin project/name to the package id)
        project_name = args.wandb_project or f"{config['dataset']}_GRDR"
        run_name = args.wandb_run_name or args.exp_name or None
        wandb.init(project=project_name, name=run_name, config=config)

        checkpoint = args.init_checkpoint
        global_step = 0
        if args.start_loop > 0 and checkpoint is None:
            raise ValueError(f'--start_loop={args.start_loop} requires --init_checkpoint')

        if args.skip_pretrain and args.init_checkpoint is None:
            raise ValueError('--skip_pretrain requires --init_checkpoint')

        # Build feature cache once for the whole chain. Each train()/test_dr() call would
        # otherwise reload the same 4 pickles (~3 min × 12 calls). load_train_text is gated
        # by the k-means cache because the only consumer of train_text is the k-means step,
        # and its result is the cache itself; if the cache exists, we never need train_text.
        train_kmeans_cached = has_kmeans_cache(
            args.dataset, 'train', args.num_latent_tokens, args.cache_dir,
            use_pseudo_queries=args.use_pseudo_queries
        )
        print(f'Pre-building feature cache for {args.dataset} (train_text load={not train_kmeans_cached})')
        feature_cache = load_shared_features(
            dataset_name=args.dataset,
            features_root=args.features_root,
            logger=print,
            use_pseudo_queries=args.use_pseudo_queries,
            load_train_text=not train_kmeans_cached,
        )
        config['feature_cache'] = feature_cache

        for loop in range(args.start_loop, args.max_length):
            # Phase 1: Pre-train
            config['loop'] = loop
            config['save_path'] = os.path.join(save_root, f'model-{loop + 1}-pre')
            config['code_length'] = loop + 1
            config['prev_model'] = checkpoint
            config['prev_id'] = f'{checkpoint}.code' if checkpoint is not None else None
            config['epochs'] = 3 if loop == 0 else args.pretrain_epochs
            config['loss_w'] = 1
            config['lr'] = args.pretrain_lr
            if loop == args.start_loop and args.skip_pretrain:
                print(f'--skip_pretrain: reusing init_checkpoint as model-{loop + 1}-pre/best_model.pt: {args.init_checkpoint}')
                checkpoint = args.init_checkpoint
            else:
                checkpoint, global_step = train(config, global_step)
            # Loop-0 pre is the only phase that consumes train_text (k-means build).
            # Drop it before any subsequent phase to match the prior call-local memory profile.
            if loop == 0 and feature_cache.get('train_text'):
                import gc
                n = len(feature_cache['train_text'])
                feature_cache['train_text'] = {}
                gc.collect()
                print(f'Dropped train_text ({n} entries) from feature_cache after loop-0 pre k-means build')
            test_dr(config, checkpoint)

            # Phase 2: Main Training
            config['save_path'] = os.path.join(save_root, f'model-{loop + 1}')
            config['prev_model'] = checkpoint
            # codebook_init must point at the kmeans.* artifact that test_dr just
            # wrote to the model-{loop+1}-pre save_path of the CURRENT run. The
            # previous form `f'{checkpoint}.kmeans.…'` resolved to the resume-source
            # checkpoint's directory (e.g. the v3/init_checkpoint path) which
            # FileNotFoundError'd on every multi-phase resume; the symlink hack we
            # used to work around it produced a coupling between unrelated runs.
            # Derive the path from the current save_root + the pre save_path that
            # test_dr writes to so it always resolves to this run's own kmeans.
            prev_save_path = os.path.join(save_root, f'model-{loop + 1}-pre')
            config['codebook_init'] = os.path.join(
                prev_save_path, f'best_model.pt.kmeans.{args.code_num}'
            )
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
