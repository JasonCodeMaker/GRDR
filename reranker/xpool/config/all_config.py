import os
import argparse
from config.base_config import Config
from modules.basic_utils import mkdirp, deletedir


class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def parse_args(self):
        description = 'Text-to-Video Retrieval'
        parser = argparse.ArgumentParser(description=description)
        
        # data parameters
        parser.add_argument('--dataset_name', type=str, default='MSRVTT', help="Dataset name")
        parser.add_argument('--videos_dir', type=str, default='dataset/msrvtt_data/MSRVTT_Videos', help="Location of videos")
        parser.add_argument('--msrvtt_train_file', type=str, default='9k')
        parser.add_argument('--panda_use_pseudo_queries', action='store_true', default=False,
                            help="PANDA only: train on panda_ret_train_addition.json (P7 pseudo-queries). Default off uses panda_ret_train.json (original 2.15M captions).")
        parser.add_argument('--panda_distractor_manifest', type=str, default=None,
                            help="PANDA only: JSON file with {'video_ids': [...]} restricting the expanded_pool train list. Used by P1.c distractor sweeps.")
        parser.add_argument('--pool_batch_size', type=int, default=64,
                            help="Text-batch size for transformer pool_frames during eval. Lower this on large expanded pools to control CPU memory (pooled_batch ~ N_vids * pool_batch_size * 512 * 4 bytes).")
        parser.add_argument('--num_frames', type=int, default=12)
        parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=int, default=224)

        # experiment parameters
        parser.add_argument('--exp_name', type=str, default='test', help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str, default='./outputs')
        parser.add_argument('--save_every', type=int, default=1, help="Save model every n epochs")
        parser.add_argument('--log_step', type=int, default=10, help="Print training log every n steps")
        parser.add_argument('--evals_per_epoch', type=int, default=5, help="Number of times to evaluate per epoch")
        parser.add_argument('--load_epoch', type=int, help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--best_r1_floor', type=float, default=-1.0, help="Seed Trainer.best so a resumed run will not overwrite model_best.pth unless R@1 exceeds this floor")
        parser.add_argument('--early_stop_patience', type=int, default=0, help="Stop training when val R@1 fails to improve for this many consecutive evals (0 = disabled)")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")
        parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")

        # model parameters
        parser.add_argument('--huggingface', action='store_true', default=True)
        parser.add_argument('--arch', type=str, default='clip_transformer')
        parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help="CLIP arch. only when not using huggingface")
        parser.add_argument('--embed_dim', type=int, default=512, help="Dimensionality of the model embedding")

        # training parameters
        parser.add_argument('--loss', type=str, default='clip')
        parser.add_argument('--clip_lr', type=float, default=1e-6, help='Learning rate used for CLIP params')
        parser.add_argument('--noclip_lr', type=float, default=1e-5, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_epochs', type=int, default=5)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')

        # frame pooling parameters
        parser.add_argument('--pooling_type', type=str)
        parser.add_argument('--k', type=int, default=-1, help='K value for topk pooling')
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=1, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--transformer_dropout', type=float, default=0.3, help='Dropout prob. in the transformer pooling')

        # candidate reranking parameters
        parser.add_argument('--eval_checkpoint', type=str, default='reranker/xpool/ckpt/msrvtt9k_model_best.pth', help='Checkpoint path for evaluation')
        parser.add_argument('--candidate_file', type=str, default=None, help='Path to candidate JSON file for reranking mode')
        parser.add_argument('--rerank_mode', action='store_true', default=False, help='Enable candidate-based evaluation')
        parser.add_argument('--index_safe_candidate_mask', action='store_true', default=False,
                            help='Build candidate masks from candidate JSON row order instead of query text')
        parser.add_argument('--save_per_query_ranks', type=str, default=None,
                            help='Optional JSON path for per-query ranks from batch evaluation')

        # expanded pool evaluation parameters
        parser.add_argument('--expanded_pool', action='store_true',
                            help='Add training videos to search pool for expanded evaluation')
        parser.add_argument('--use_cached_video_features', action='store_true',
                            help='Load all evaluation video features from cache instead of extracting them from media files')
        parser.add_argument('--video_cache_dir', type=str, default=None,
                            help='Base directory containing cached video features; defaults to the architecture-specific cache root')

        # result saving parameters
        parser.add_argument('--result_file', type=str, default='test_results.csv', help='Filename for CSV results (saved to output/evaluation_results/rerank/)')

        # system parameters
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=24, help='Random seed')
        parser.add_argument('--no_tensorboard', action='store_true', default=False)
        parser.add_argument('--tb_log_dir', type=str, default='logs')

        args, _ = parser.parse_known_args()

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args
