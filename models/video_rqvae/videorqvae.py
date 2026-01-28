import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer

# Import encoder and decoder classes from dedicated files
from .encoder import VideoLatentEncoder_V2
from .decoder import VideoLatentDecoder_V2

class VideoRQVAE_V2(nn.Module):
    def __init__(self,
                 in_dim=512,
                 num_latent_tokens=4,
                 num_emb_list=[256, 256, 256, 256],
                 e_dim=64,
                 mlp_layers=None,             # Hidden layers for encoder/decoder MLPs
                 dropout_prob=0.0,            # MLP dropout probability
                 bn=False,                    # Whether to use batch norm inside MLPs
                 quant_loss_weight=1.0,       # Weight for quantization loss
                 kmeans_init=True,            # Use k-means initialization for codebook
                 kmeans_iters=100,            # Number of k-means iterations
                 sk_epsilons=0.0,            # Sinkhorn epsilon values for each RQ level
                 sk_iters=100,                # Sinkhorn iterations
                 use_linear=0,                # Use linear projection in RQ
                 beta=0.65,                   # Beta weight for VQ commitment loss
                 diversity_loss_weight=0.0,   # Weight for latent token diversity loss
                 contrastive_temperature=0.07,  # Temperature for InfoNCE contrastive loss
                 vid_loss_weight=None,        # Array [mse, l1, cosine] weights
                 text_alignment_dim=None       # Output dimension for text alignment projector
        ):
        super(VideoRQVAE_V2, self).__init__()

        # Persist configuration for checkpointing / recreation
        self.config = {
            'in_dim': in_dim,
            'num_latent_tokens': num_latent_tokens,
            'num_emb_list': num_emb_list,
            'e_dim': e_dim,
            'mlp_layers': mlp_layers,
            'dropout_prob': dropout_prob,
            'bn': bn,
            'quant_loss_weight': quant_loss_weight,
            'kmeans_init': kmeans_init,
            'kmeans_iters': kmeans_iters,
            'sk_epsilons': sk_epsilons,
            'sk_iters': sk_iters,
            'use_linear': use_linear,
            'beta': beta,
            'diversity_loss_weight': diversity_loss_weight,
            'contrastive_temperature': contrastive_temperature,
            'vid_loss_weight': vid_loss_weight,
            'text_alignment_dim': text_alignment_dim,
        }

        # Store attributes used in class methods
        self.num_latent_tokens = num_latent_tokens
        self.quant_loss_weight = quant_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.contrastive_temperature = contrastive_temperature

        # Default vid_loss_weight: [mse, l1, cosine, cls, p2p_infonce]
        # Set default to use only MSE loss (index 0) if not specified
        self.vid_loss_weight = vid_loss_weight if vid_loss_weight is not None else [1.0, 0.0, 0.0, 0.0, 0.0]

        # Build MLP layer dimensions
        # If mlp_layers not specified, use direct mapping [in_dim, e_dim]
        if mlp_layers is None:
            self.encode_layer_dims = [in_dim, e_dim]
        else:
            self.encode_layer_dims = [in_dim] + mlp_layers + [e_dim]
        self.encoder = VideoLatentEncoder_V2(
            input_dim=in_dim,
            num_latent_tokens=num_latent_tokens,
            mlp_layers=self.encode_layer_dims,
            mlp_dropout=dropout_prob,
            mlp_bn=bn
        )

        # ResidualVectorQuantizer for quantizing latent tokens
        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
            use_linear=use_linear,
            beta=beta
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = VideoLatentDecoder_V2(
            output_dim=in_dim,
            num_latent_tokens=num_latent_tokens,
            mlp_layers=self.decode_layer_dims,
            mlp_dropout=dropout_prob,
            mlp_bn=bn
        )

        # Text alignment projector: maps decoded features to text embedding space
        # If text_alignment_dim is None or 0, disable the projector (use x_decoded directly)
        if text_alignment_dim is None or text_alignment_dim == 0:
            self.text_alignment_projector = None
            self.config['text_alignment_dim'] = None
        else:
            self.text_alignment_projector = nn.Linear(e_dim, text_alignment_dim)
            # Initialize with Xavier uniform
            nn.init.xavier_uniform_(self.text_alignment_projector.weight)
            nn.init.zeros_(self.text_alignment_projector.bias)

    def get_config(self):
        """
        Return model configuration for checkpointing.
        """
        return self.config.copy()

    @classmethod
    def from_config(cls, config, **override_params):
        """
        Reconstruct VideoRQVAE_V2 from configuration dict.
        """
        merged_config = {**config, **override_params}

        # Filter out metadata fields that aren't model parameters
        metadata_keys = {'feature_extractor'}  # Add other metadata keys here as needed
        filtered_config = {k: v for k, v in merged_config.items() if k not in metadata_keys}

        return cls(**filtered_config)

    def forward(self, video_patches, text_embs=None, use_sk=True, return_probs=False):
        """
        Forward pass for InternVideo2 pooled video features.

        Args:
            video_patches: [batch_size, in_dim] - Single pooled video feature per video
            text_embs: Optional [batch_size, num_texts, text_dim] for contrastive learning
            use_sk: Whether to use Sinkhorn for quantization
            return_probs: Whether to return raw distance distributions over codebook

        Returns:
            reconstructed: [batch_size, in_dim] - Reconstructed video feature
            rq_loss: Quantization loss
            indices: Quantization indices
            x_encoded: Encoded features before quantization
            x_decoded: [batch_size, num_latent_tokens, e_dim] - Decoded features before projection
            x_aligned: [batch_size, num_latent_tokens, D] - Projected features for contrastive loss
                      (D = text_alignment_dim if projector enabled, else D = e_dim)
            raw_distances: (Optional) List of [batch_size, num_latent_tokens, codebook_size] raw distance tensors per RQ layer
                          Only returned if return_probs=True. Length = number of RQ layers.
        """
        # Encode pooled video features into latent slots
        x_encoded = self.encoder(video_patches)

        # Quantize latent tokens: [batch, num_latent_tokens, e_dim]
        q_video_emb, rq_loss, indices, distances, residuals = self.rq(
            x_encoded, use_sk=use_sk, return_probs=return_probs
        )

        # VideoDecoder reconstructs video feature from quantized latent tokens
        # x_q: [batch, num_latent_tokens, e_dim] -> reconstructed: [batch, in_dim]
        x_decoded, reconstructed = self.decoder(q_video_emb)

        # Project decoded features to text embedding space for contrastive loss
        # If text_alignment_projector is disabled, use x_decoded directly
        if self.text_alignment_projector is not None:
            x_aligned = self.text_alignment_projector(x_decoded)  # [batch, num_latent_tokens, text_alignment_dim]
        else:
            x_aligned = x_decoded  # [batch, num_latent_tokens, e_dim]

        if return_probs:
            return reconstructed, rq_loss, indices, x_encoded, x_decoded, x_aligned, distances
        else:
            return reconstructed, rq_loss, indices, x_encoded, x_decoded, x_aligned

    @torch.no_grad()
    def get_indices(self, video_patches, use_sk=False, return_quantized_features=False):
        """
        Get quantization indices for InternVideo2 pooled features.

        Args:
            video_patches: [batch_size, in_dim] - Single pooled video feature per video
            use_sk: Whether to use Sinkhorn for quantization
            return_quantized_features: bool - Whether to return quantized embeddings for sID feature extraction

        Returns:
            indices: Quantization indices [batch_size, num_latent_tokens, num_rq_layers]
            quantized_features: Optional[torch.Tensor] - Quantized embeddings [batch_size, num_latent_tokens, e_dim]
                               (only when return_quantized_features=True)
        """
        # Encode pooled video features into latent slots
        x_encoded = self.encoder(video_patches)

        # Quantize latent tokens: [batch, num_latent_tokens, e_dim]
        # Note: return_probs=False, so 4th return is None
        q_video_emb, rq_loss, indices, distances, residuals = self.rq(x_encoded, use_sk=use_sk, return_probs=False)

        if return_quantized_features:
            return indices, q_video_emb
        return indices


    def compute_contrastive_loss(self, x_decoded, text_embs, text_group_ids=None):
        """
        Compute contrastive loss between video and text embeddings.

        Two modes:
        1. Training mode (text_group_ids provided): Multi-positive group-based InfoNCE loss
        2. Test mode (text_group_ids=None): Simple 1-to-1 best-token InfoNCE loss

        Args:
            x_decoded: [batch_size, num_latent_tokens, embed_dim] - Decoded video embeddings
                      (should be x_aligned from forward() if text_alignment_projector is used)
            text_embs: [batch_size, num_texts, text_dim] or [batch_size, text_dim] - Text embeddings
            text_group_ids: [batch_size, num_texts] - Optional group assignment (training only)

        Returns:
            contrastive_loss: Scalar tensor - InfoNCE contrastive loss
        """
        batch_size, num_tokens, video_dim = x_decoded.shape
        device = x_decoded.device

        # ============ Test Mode: Simple 1-to-1 Best-Token Matching ============
        if text_group_ids is None:
            # Handle text_embs shape - squeeze if [B, 1, text_dim]
            if text_embs.dim() == 3 and text_embs.shape[1] == 1:
                text_embs = text_embs.squeeze(1)  # [B, 1, text_dim] -> [B, text_dim]

            text_dim = text_embs.shape[-1]
            
            # Validate dimension compatibility
            if video_dim != text_dim:
                raise ValueError(
                    f"Dimension mismatch in compute_contrastive_loss: "
                    f"video embeddings have dim={video_dim}, text embeddings have dim={text_dim}. "
                    f"When using text_alignment_projector, ensure text_alignment_dim matches text embedding dimension, "
                    f"or pass x_aligned (6th return value from forward()) instead of x_decoded."
                )

            # Normalize embeddings for cosine similarity
            video_tokens_norm = F.normalize(x_decoded, dim=-1, eps=1e-12)  # [B, num_tokens, video_dim]
            text_norm = F.normalize(text_embs, dim=-1, eps=1e-12)  # [B, text_dim]

            # Compute similarity between each video's tokens and its corresponding text
            text_expanded = text_norm.unsqueeze(1)  # [B, 1, text_dim]
            per_video_similarity = torch.sum(video_tokens_norm * text_expanded, dim=-1)  # [B, num_tokens]

            # Select token with highest similarity for each video
            best_token_indices = torch.argmax(per_video_similarity, dim=1)  # [B]
            batch_indices = torch.arange(batch_size, device=device)
            selected_video_emb = x_decoded[batch_indices, best_token_indices]  # [B, video_dim]

            # Normalize selected embeddings
            selected_video_norm = F.normalize(selected_video_emb, dim=-1, eps=1e-12)  # [B, video_dim]

            # Build [batch_size, batch_size] similarity matrix and compute InfoNCE loss
            similarity_matrix = torch.matmul(selected_video_norm, text_norm.T)  # [B, B]
            similarity_with_temp = similarity_matrix / self.contrastive_temperature
            labels = torch.arange(batch_size, device=device)
            contrastive_loss = F.cross_entropy(similarity_with_temp.T, labels, reduction='mean')

            return contrastive_loss

        # ============ Training Mode: Multi-Positive Group-Based Loss ============
        _, num_texts, text_dim = text_embs.shape
        
        # Validate dimension compatibility
        if video_dim != text_dim:
            raise ValueError(
                f"Dimension mismatch in compute_contrastive_loss: "
                f"video embeddings have dim={video_dim}, text embeddings have dim={text_dim}. "
                f"When using text_alignment_projector, ensure text_alignment_dim matches text embedding dimension, "
                f"or pass x_aligned (6th return value from forward()) instead of x_decoded."
            )

        # ============ Flatten for global contrastive learning ============
        # Reshape: [B, N, D] -> [B*N, D]
        video_tokens = x_decoded.reshape(batch_size * num_tokens, video_dim)  # [B*num_tokens, video_dim]
        text_tokens = text_embs.reshape(batch_size * num_texts, text_dim)    # [B*num_texts, text_dim]

        # L2 normalize embeddings for cosine similarity
        video_tokens_norm = F.normalize(video_tokens, dim=-1, eps=1e-12)
        text_tokens_norm = F.normalize(text_tokens, dim=-1, eps=1e-12)

        # Compute similarity matrix: [B*num_tokens, B*num_texts]
        similarity_matrix = torch.matmul(video_tokens_norm, text_tokens_norm.T) / self.contrastive_temperature

        # ============ Create ID mappings for alignment ============
        # Video IDs: [0,0,0,0, 1,1,1,1, ..., B-1,B-1,B-1,B-1] for num_tokens=4
        video_ids_v = torch.arange(batch_size, device=device).repeat_interleave(num_tokens)  # [B*num_tokens]
        video_ids_t = torch.arange(batch_size, device=device).repeat_interleave(num_texts)   # [B*num_texts]

        # Token IDs within each video: [0,1,2,3, 0,1,2,3, ..., 0,1,2,3]
        token_ids = torch.arange(num_tokens, device=device).repeat(batch_size)  # [B*num_tokens]

        # Flatten text_group_ids: [B, num_texts] -> [B*num_texts]
        group_ids_flat = text_group_ids.reshape(batch_size * num_texts)  # [B*num_texts]

        # ============ Video-to-Text Direction ============
        # For each video token i, find ALL texts from SAME VIDEO with group_id == i

        # Same video mask: [B*num_tokens, B*num_texts]
        same_video_v2t = video_ids_v.unsqueeze(1) == video_ids_t.unsqueeze(0)

        # Group alignment mask: video token i (with token_id=k) matches texts with group_id=k
        # [B*num_tokens, B*num_texts]
        group_match_v2t = token_ids.unsqueeze(1) == group_ids_flat.unsqueeze(0)

        # Positive mask: same video AND matching group (multi-positive)
        positive_mask_v2t = same_video_v2t & group_match_v2t  # [B*num_tokens, B*num_texts]

        # Negative mask: different video
        negative_mask_v2t = ~same_video_v2t  # [B*num_tokens, B*num_texts]

        # ============ Multi-Positive InfoNCE (video→text) ============
        # Numerator: log(Σ_p exp(sim(i,p))) over all positives
        positive_sims_v2t = similarity_matrix.clone()
        positive_sims_v2t[~positive_mask_v2t] = float('-inf')
        log_sum_pos_v2t = torch.logsumexp(positive_sims_v2t, dim=1)  # [B*num_tokens]

        # Denominator: log(Σ_j exp(sim(i,j))) over positives + negatives
        valid_mask_v2t = positive_mask_v2t | negative_mask_v2t
        denominator_sims_v2t = similarity_matrix.clone()
        denominator_sims_v2t[~valid_mask_v2t] = float('-inf')
        log_sum_all_v2t = torch.logsumexp(denominator_sims_v2t, dim=1)  # [B*num_tokens]

        # InfoNCE: -log(numerator/denominator) = -(log_num - log_denom)
        loss_v2t_per_token = -log_sum_pos_v2t + log_sum_all_v2t  # [B*num_tokens]

        # Handle edge case: tokens with no positives (filter out -inf)
        valid_tokens_v2t = torch.isfinite(log_sum_pos_v2t)
        if valid_tokens_v2t.sum() == 0:
            loss_v2t = torch.tensor(0.0, device=device)
        else:
            loss_v2t = loss_v2t_per_token[valid_tokens_v2t].mean()

        # ============ Text-to-Video Direction (Symmetric) ============
        # For each text with group_id=k, find video token k from SAME VIDEO

        # Same video mask: [B*num_texts, B*num_tokens]
        same_video_t2v = video_ids_t.unsqueeze(1) == video_ids_v.unsqueeze(0)

        # Group alignment mask: text with group_id=k matches video token k
        # [B*num_texts, B*num_tokens]
        group_match_t2v = group_ids_flat.unsqueeze(1) == token_ids.unsqueeze(0)

        # Positive mask: same video AND matching group
        positive_mask_t2v = same_video_t2v & group_match_t2v  # [B*num_texts, B*num_tokens]

        # Negative mask: different video
        negative_mask_t2v = ~same_video_t2v  # [B*num_texts, B*num_tokens]

        # Transpose similarity matrix for text-to-video
        similarity_matrix_t2v = similarity_matrix.T  # [B*num_texts, B*num_tokens]

        # ============ Multi-Positive InfoNCE (text→video) ============
        # Numerator: log(Σ_p exp(sim(i,p))) over all positives
        positive_sims_t2v = similarity_matrix_t2v.clone()
        positive_sims_t2v[~positive_mask_t2v] = float('-inf')
        log_sum_pos_t2v = torch.logsumexp(positive_sims_t2v, dim=1)  # [B*num_texts]

        # Denominator: log(Σ_j exp(sim(i,j))) over positives + negatives
        valid_mask_t2v = positive_mask_t2v | negative_mask_t2v
        denominator_sims_t2v = similarity_matrix_t2v.clone()
        denominator_sims_t2v[~valid_mask_t2v] = float('-inf')
        log_sum_all_t2v = torch.logsumexp(denominator_sims_t2v, dim=1)  # [B*num_texts]

        # InfoNCE: -log(numerator/denominator)
        loss_t2v_per_text = -log_sum_pos_t2v + log_sum_all_t2v  # [B*num_texts]

        # Handle edge case: texts with no positives
        valid_texts_t2v = torch.isfinite(log_sum_pos_t2v)
        if valid_texts_t2v.sum() == 0:
            loss_t2v = torch.tensor(0.0, device=device)
        else:
            loss_t2v = loss_t2v_per_text[valid_texts_t2v].mean()

        # ============ Final Symmetric Loss ============
        contrastive_loss = (loss_v2t + loss_t2v) / 2

        return contrastive_loss

    def compute_loss(self, recon_video_features, quant_loss, video_features=None, encoder_out=None):
        loss_recon = 0.0

        # L2 normalize video features
        video_features = F.normalize(video_features, p=2, dim=-1, eps=1e-12)
        recon_video_features = F.normalize(recon_video_features, p=2, dim=-1, eps=1e-12)

        # MSE Loss (weight index 0)
        if self.vid_loss_weight[0] > 0:
            loss_mse = F.mse_loss(recon_video_features, video_features, reduction='mean')
            loss_recon += self.vid_loss_weight[0] * loss_mse

        # Cosine Loss (weight index 2)
        if self.vid_loss_weight[2] > 0:
            loss_cosine = (1 - F.cosine_similarity(recon_video_features, video_features, dim=-1)).mean()
            loss_recon += self.vid_loss_weight[2] * loss_cosine

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon
