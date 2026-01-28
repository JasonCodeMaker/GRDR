"""
Trainer package for GRDR model.

Contains training and evaluation components.
"""

from .trainer import OurTrainer, train, build_loss_weights
from .evaluator import (
    test,
    eval_retrieval,
    save_code,
    our_encode_dual,
    build_index,
    do_retrieval,
    do_maxsim_retrieval,
    summarize_recall,
    kmeans,
    skl_kmeans,
    constrained_km,
    test_dr,
    do_epoch_encode,
    compute_sid_collision_stats,
    compute_train_test_collision,
    build_sid_to_videos_mapping,
    balance,
    conflict,
    norm_by_prefix,
)

__all__ = [
    # trainer.py
    'OurTrainer',
    'train',
    'build_loss_weights',
    # evaluator.py
    'test',
    'eval_retrieval',
    'save_code',
    'our_encode_dual',
    'build_index',
    'do_retrieval',
    'do_maxsim_retrieval',
    'summarize_recall',
    'kmeans',
    'skl_kmeans',
    'constrained_km',
    'test_dr',
    'do_epoch_encode',
    'compute_sid_collision_stats',
    'compute_train_test_collision',
    'build_sid_to_videos_mapping',
    'balance',
    'conflict',
    'norm_by_prefix',
]
