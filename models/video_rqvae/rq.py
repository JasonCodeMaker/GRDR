import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100, use_linear=0, beta=0.55):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = [sk_epsilons] * self.num_quantizers
        self.sk_iters = sk_iters

        # Build quantizer layers
        self.vq_layers = nn.ModuleList()
        for n_e, sk_epsilon in zip(self.n_e_list, self.sk_epsilons):
            quantizer = VectorQuantizer(
                n_e, e_dim,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters,
                use_linear=use_linear,
                beta=beta
            )
            self.vq_layers.append(quantizer)

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True, return_probs=False, temperature=None):
        all_losses = []
        all_indices = []
        all_distances = [] if return_probs else None
        residuals = []

        x_q = 0
        residual = x

        for quantizer in self.vq_layers:
            # Normalize residual before quantization to ensure consistent cosine distance computation
            # Each quantizer receives unit-normalized input for proper similarity scoring
            # The returned x_res is unnormalized (from codebook.weight), keeping residual arithmetic consistent
            residual_normalized = F.normalize(residual, dim=-1, eps=1e-12)

            # Temperature parameter is deprecated - no longer used for softmax in VQ
            # Kept in signature for backward compatibility but ignored
            x_res, loss, indices, distances_per_layer = quantizer(
                residual_normalized, use_sk=use_sk, return_probs=return_probs, temperature=None
            )
            # Residual subtraction stays in unnormalized space for arithmetic consistency
            residual = residual - x_res.detach()
            x_q = x_q + x_res

            residuals.append(residual)
            all_losses.append(loss)
            all_indices.append(indices)
            if return_probs:
                all_distances.append(distances_per_layer)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices, all_distances, residuals

    def forward_progressive(self, x, current_layer, cached_cumulative=None,
                            cached_indices=None, use_sk=True, return_probs=False):
        """
        Progressive forward: use cached cumulative for previous layers.

        This method supports progressive layer-by-layer training where:
        - Previous layer codes remain FIXED (using cached cumulative quantization)
        - Encoder can train freely (gradients flow through current layer)
        - T5 supervision remains valid (no code drift)

        Args:
            x: Live encoder output [B, e_dim] for selected token
            current_layer: Which layer to train (0-indexed)
            cached_cumulative: [B, e_dim] sum of x_res from layers 0..current_layer-1
            cached_indices: List of [B] tensors for layers 0..current_layer-1
            use_sk: Use Sinkhorn for current layer
            return_probs: Return distance probabilities

        Returns:
            x_q: Total quantized output (cached + new layer)
            loss: Quantization loss (only from current layer)
            all_indices: Indices for all layers [B, current_layer+1]
            all_distances: Distance probabilities if return_probs=True
            residuals: Residual after current layer
        """
        all_indices = []
        all_distances = [] if return_probs else None

        if current_layer == 0:
            # First layer: standard forward (no cache)
            residual = x
            x_q = torch.zeros_like(x)
        else:
            # Use cached cumulative (detached to prevent gradient flow to previous layers)
            # Compute fresh residual from live encoder output
            residual = x - cached_cumulative.detach()
            x_q = cached_cumulative.detach()

            # Add cached indices for previous layers (as tensors)
            for cached_idx in cached_indices:
                all_indices.append(cached_idx)

        # Quantize only the current layer
        residual_norm = F.normalize(residual, dim=-1, eps=1e-12)
        x_res, loss, indices, distances = self.vq_layers[current_layer](
            residual_norm, use_sk=use_sk, return_probs=return_probs
        )

        x_q = x_q + x_res
        all_indices.append(indices)
        if return_probs:
            all_distances.append(distances)

        # Stack indices: [B, current_layer+1]
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, loss, all_indices, all_distances, [residual - x_res.detach()]