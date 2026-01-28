import glob
import json
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.optim import AdamW
from peft import TaskType, LoraConfig, get_peft_model
from transformers.trainer_pt_utils import get_parameter_names
from tqdm import tqdm


def seed_everything(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def sinkhorn_raw(out: Tensor, epsilon: float,
                 sinkhorn_iterations: int,
                 use_distrib_train: bool):
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper

    B = Q.shape[1]
    K = Q.shape[0]  # how many prototypes
    # make the matrix sums to 1
    sum_Q = torch.clamp(torch.sum(Q), min=1e-5)
    if use_distrib_train:
        B *= dist.get_world_size()
        dist.all_reduce(sum_Q)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.clamp(torch.sum(Q, dim=1, keepdim=True), min=1e-5)
        if use_distrib_train:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.clamp(torch.sum(torch.sum(Q, dim=0, keepdim=True), dim=1, keepdim=True), min=1e-5)
        Q /= B
    Q *= B
    return Q.t()


def get_optimizer(model, lr, code_length=None, encoder_lr_scale=1.0):
    """
    Create optimizer with special handling for different parameter groups.

    Args:
        model: Model instance
        lr: Base learning rate
        code_length: Number of RQ layers (if None, extracted from model)
        encoder_lr_scale: Learning rate multiplier for VideoRQVAE encoder (default 0.6)
    """
    if code_length is None:
        code_length = model.code_length if hasattr(model, 'code_length') else 1

    # Target the last codebook layer for 10× learning rate
    last_codebook_param_name = f"code_embedding.{code_length - 1}.weight"

    # Target VideoRQVAE encoder for reduced learning rate (60% of base)
    encoder_param_prefix = "video_rqvae.encoder."

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    # Helper to check if param belongs to encoder
    def is_encoder_param(name):
        return name.startswith(encoder_param_prefix)

    # Helper to check if param is last codebook
    def is_last_codebook(name):
        return last_codebook_param_name in name

    optimizer_grouped_parameters = [
        {
            # VideoRQVAE encoder parameters with reduced LR (60%)
            "params": [p for n, p in model.named_parameters()
                      if p.requires_grad and is_encoder_param(n)],
            "weight_decay": 0.0,
            "lr": lr * encoder_lr_scale
        },
        {
            # Last codebook layer gets 10× higher learning rate
            "params": [p for n, p in model.named_parameters()
                      if p.requires_grad and is_last_codebook(n) and not is_encoder_param(n)],
            "weight_decay": 0.0,
            "lr": lr * 10
        },
        {
            # Decay parameters (excluding encoder and last codebook)
            "params": [p for n, p in model.named_parameters()
                      if p.requires_grad and n in decay_parameters
                      and not is_encoder_param(n) and not is_last_codebook(n)],
            "weight_decay": 0.0,
        },
        {
            # Non-decay parameters (excluding encoder and last codebook)
            "params": [p for n, p in model.named_parameters()
                      if p.requires_grad and n not in decay_parameters
                      and not is_encoder_param(n) and not is_last_codebook(n)],
            "weight_decay": 0.0,
        },
    ]

    # Filter out empty parameter groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]

    optimizer = AdamW(optimizer_grouped_parameters, lr)
    return optimizer


def compute_model_stats(model):
    """
    Compute model parameter statistics.

    Args:
        model: PyTorch model

    Returns:
        dict with 'total', 'trainable', 'frozen' parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


class Tree:
    """
    Tree structure for constrained generation in beam search.

    This tree stores valid code sequences and provides prefix-based lookup
    to constrain the generation to only valid semantic IDs.
    """

    def __init__(self):
        self.root = dict()

    def set(self, path):
        """Add a valid path to the tree."""
        pointer = self.root
        for i in path:
            if i not in pointer:
                pointer[i] = dict()
            pointer = pointer[i]

    def set_all(self, path_list):
        """Add multiple paths to the tree."""
        for path in tqdm(path_list):
            self.set(path)

    def find(self, path):
        """Find valid next tokens given a prefix path."""
        if isinstance(path, torch.Tensor):
            path = path.cpu().tolist()
        pointer = self.root
        for i in path:
            if i not in pointer:
                return []
            pointer = pointer[i]
        return list(pointer.keys())

    def __call__(self, batch_id, path):
        """Callable interface for beam search prefix_allowed_tokens_fn."""
        return self.find(path)


class CodeDriftMonitor:
    """
    Lightweight monitor for tracking code drift during progressive RQ-VAE training.

    Detects when encoder updates cause previous layer codes to change, which
    undermines T5's semantic ID learning that depends on consistent codes.
    """

    def __init__(self, num_layers: int, token_idx_mapping: dict = None):
        self.num_layers = num_layers
        self.baseline_codes = {}  # {sample_key: [layer_0_code, layer_1_code, ...]}
        self.current_layer = 0    # Which layer is being trained
        self.token_idx_mapping = token_idx_mapping or {}  # Maps sample_key to assigned token index

    def snapshot_baseline(self, sample_codes_dict: dict):
        """
        Store baseline codes at loop start.

        Args:
            sample_codes_dict: Dict[sample_key -> codes] where codes can be:
                - Flat format [c0, c1, c2] from json file (best_model.pt.code)
                - Nested format [[c0,c1,c2], [c0,c1,c2]] from gen_sid() (per-token)
        """
        self.baseline_codes = {}
        for k, v in sample_codes_dict.items():
            # Handle both flat and nested formats
            if v and isinstance(v[0], list):
                # Nested format from gen_sid(): take first token's codes
                codes = v[0][:self.current_layer]
            else:
                # Flat format from json file
                codes = v[:self.current_layer]
            self.baseline_codes[k] = codes
        return len(self.baseline_codes)

    def compute_drift(self, current_codes_dict: dict) -> dict:
        """
        Compute drift metrics comparing current codes to baseline.

        Args:
            current_codes_dict: Dict[sample_key -> codes] where codes can be:
                - Flat format [c0, c1, c2] from json file
                - Nested format [[c0,c1,c2], [c0,c1,c2]] from gen_sid() (per-token)

        Returns:
            dict with drift metrics
        """
        if not self.baseline_codes or self.current_layer == 0:
            return {
                'drift_rate_total': 0.0,
                'drift_rate_per_layer': [],
                'drifted_count': 0,
                'total_samples': len(self.baseline_codes),
                'drifted_samples': []
            }

        total_samples = len(self.baseline_codes)
        layer_drifts = [0] * self.current_layer
        any_drift = 0
        drifted_samples = []

        for sample_key, baseline_codes in self.baseline_codes.items():
            if sample_key not in current_codes_dict:
                continue

            current = current_codes_dict[sample_key]
            # Handle both flat and nested formats
            if current and isinstance(current[0], list):
                # Nested format from gen_sid(): use assigned token
                token_idx = self.token_idx_mapping.get(sample_key, 0)
                current_codes = current[token_idx][:self.current_layer]
            else:
                # Flat format
                current_codes = current[:self.current_layer]

            sample_drifted = False
            for layer_idx, (base, curr) in enumerate(zip(baseline_codes, current_codes)):
                if base != curr:
                    layer_drifts[layer_idx] += 1
                    sample_drifted = True

            if sample_drifted:
                any_drift += 1
                drifted_samples.append(sample_key)

        return {
            'drift_rate_total': any_drift / total_samples * 100 if total_samples > 0 else 0.0,
            'drift_rate_per_layer': [d / total_samples * 100 if total_samples > 0 else 0.0 for d in layer_drifts],
            'drifted_count': any_drift,
            'total_samples': total_samples,
            'drifted_samples': drifted_samples[:10]  # First 10 for debugging
        }


def create_videorqvae(
    code_num=256,
    code_length=4,
    num_latent_tokens=1,
    e_dim=64,
    in_dim=512,
    device=0
):
    """
    Create an initialized VideoRQVAE model without loading from checkpoint.

    Args:
        code_num: Size of each codebook (default 256)
        code_length: Number of RQ layers (default 4)
        num_latent_tokens: Number of latent tokens (default 1)
        e_dim: Embedding dimension for codebook (default 64)
        in_dim: Input dimension (default 512 for InternVideo2 features)
        device: CUDA device index, torch.device object, or device string

    Returns:
        model: Initialized VideoRQVAE_V2 model in eval mode
    """
    # Lazy import to avoid circular dependencies
    from models.video_rqvae.videorqvae import VideoRQVAE_V2

    logger = logging.getLogger(__name__)

    logger.info(f"Creating new VideoRQVAE model (code_num={code_num}, code_length={code_length})")

    # Build num_emb_list for each RQ layer
    num_emb_list = [code_num] * code_length

    model = VideoRQVAE_V2(
        in_dim=in_dim,
        num_latent_tokens=num_latent_tokens,
        num_emb_list=num_emb_list,
        e_dim=e_dim,
        kmeans_init=False,  # No k-means init since we're starting fresh
    )

    # Normalize device to string format
    if isinstance(device, torch.device):
        device_str = str(device)
    elif isinstance(device, str):
        device_str = device
    else:
        device_str = f'cuda:{device}'

    # Move to device and set to eval mode
    model = model.to(device_str)
    model.eval()

    logger.info(f"VideoRQVAE model created successfully (device: {device_str})")

    return model

