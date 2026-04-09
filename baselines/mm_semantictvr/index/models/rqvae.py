import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer

class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 text_loss_type="mse",
                 text_loss_pos="before",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
                 # EMA parameters
                 use_ema=False,
                 ema_decay=0.99,
                 beta=0.55
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.text_loss_type = text_loss_type
        self.text_loss_pos = text_loss_pos
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear,
                                          beta=beta,
                                          ema_update=use_ema,
                                          decay=ema_decay)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        x_encoded = self.encoder(x)
        x_q, rq_loss, indices, distances = self.rq(x_encoded, use_sk=use_sk)
        # print(indices.shape)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_encoded

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        return indices, distances

    def compute_loss(self, out, encoder_out, quant_loss, xs, cap=None):
        
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        # Text loss computation
        if self.text_loss_pos == 'before':
            output = encoder_out
        elif self.text_loss_pos == 'after':
            output = out
        else:
            raise ValueError(f'Invalid text_loss_pos: {self.text_loss_pos}. Must be "before" or "after"')

        if self.text_loss_type == 'mse':
            loss_cap = F.mse_loss(output, cap, reduction='mean') if cap is not None else torch.tensor(0.0, device=output.device)
        elif self.text_loss_type == 'l1':
            loss_cap = F.l1_loss(output, cap, reduction='mean') if cap is not None else torch.tensor(0.0, device=output.device)
        elif self.text_loss_type == 'contrastive':
            if cap is not None:
                eps = 1e-12
                normalized_out = output / (output.norm(p=2, dim=1, keepdim=True) + eps)
                normalized_cap = cap / (cap.norm(p=2, dim=1, keepdim=True) + eps)
                logits_out = torch.matmul(normalized_out, normalized_cap.t())
                loss_1 = nn.functional.cross_entropy(logits_out, torch.arange(len(logits_out)).to(logits_out.device))
                loss_2 = nn.functional.cross_entropy(logits_out.t(), torch.arange(len(logits_out)).to(logits_out.device))
                loss_cap = (loss_1 + loss_2) / 2
            else:
                loss_cap = torch.tensor(0.0, device=output.device)
        else:
            raise ValueError(f'Invalid text_loss_type: {self.text_loss_type}. Must be "mse", "l1", or "contrastive"')

        # Calculate total loss
        if cap is None:
            loss_total = loss_recon + self.quant_loss_weight * quant_loss
            return loss_total, loss_recon
        else:
            loss_total = loss_cap + loss_recon + self.quant_loss_weight * quant_loss
            return loss_total, loss_recon, loss_cap
    
    def compute_loss_simple(self, out, quant_loss, xs=None):
        """Backward compatibility method for simple loss computation without text loss"""
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon


class VideoRQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,                    # Updated: InternVL feature dimension per patch
                 num_patches=16,               # Updated: InternVL typically has 16 patches
                 num_latent_tokens=1,
                 encoder_width=768,
                 encoder_layers=6,
                 encoder_heads=12,
                 num_emb_list=None,
                 e_dim=64,
                 text_loss_type="mse",
                 text_loss_pos="before",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
                 dropout_prob=0.0,            # Added: for backward compatibility
                 bn=False,                    # Added: for backward compatibility
                 # EMA parameters
                 use_ema=False,               # Use EMA for codebook updates instead of gradients
                 ema_decay=0.99,              # Decay rate for EMA updates
                 # TextDecoder parameters
                 use_text_decoder=False,      # Whether to include text decoder
                 text_dim=4096,               # Target text embedding dimension
                 text_decoder_layers=None,    # Hidden layers for text decoder
                 # Router parameters
                 router_hidden_dim=768,       # Hidden dimension for router
                 router_temperature=1.5,      # Temperature for router similarity computation
                 soft_temperature=1.0,        # Temperature for soft mixture control
                 # Text reconstruction loss parameter
                 text_recon_loss_weight=1.0,  # Weight for text reconstruction loss
                 # Router diversity loss parameter (optional, disabled if num_latent_tokens=1)
                 diversity_loss_weight=0.1,   # Weight for router diversity loss
                 # Contrastive loss parameters
                 contrastive_temperature=0.07,  # Temperature for InfoNCE contrastive loss
                 contrastive_loss_weight=1.0,  # Weight for contrastive loss
                 beta=0.55,  # Beta for vq loss
                 version=1.0,  # Keep for experiment naming only (not used for selection)
                 num_iterations=3,  # Number of slot attention iterations (for VideoSlotEncoder)
                 # Encoder/decoder selection
                 encoder="VideoEncoder",
                 decoder="VideoDecoder",
                 # Frame-level classification loss parameters
                 num_frames=8,  # Number of frames per video (required for loss_type='cls')
                 frame_temperature=0.1,  # Temperature for frame classification softmax
                 # Flexible video reconstruction loss weighting
                 vid_loss_weight=None  # Array [mse, l1, cosine, cls, p2p_infonce] weights; None = auto-convert from loss_type
        ):
        super(VideoRQVAE, self).__init__()

        # Store complete configuration BEFORE importing classes (captures only __init__ params)
        self.config = locals().copy()
        self.config.pop('self')
        self.config.pop('__class__', None)

        # Import encoder and decoder classes from dedicated files
        from .encoder import VideoEncoder, VideoLatentEncoder, VideoSlotEncoder
        from .decoder import VideoDecoder, VideoSlotMLPDecoder, VideoLatentDecoder

        self.in_dim = in_dim                    # Input dimension per patch (from InternVL)
        self.input_dim = in_dim                 # For backward compatibility
        self.num_patches = num_patches
        self.num_latent_tokens = num_latent_tokens
        self.encoder_width = encoder_width
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.vid_loss_weight = vid_loss_weight  
        self.text_loss_type = text_loss_type
        self.text_loss_pos = text_loss_pos
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.beta = beta
        self.num_iterations = num_iterations
        self.version = version  

        # Store backward compatibility parameters
        self.dropout_prob = dropout_prob
        self.bn = bn

        # Store EMA parameters
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Store text decoder parameters
        self.use_text_decoder = use_text_decoder
        self.text_dim = text_dim
        self.text_decoder_layers = text_decoder_layers
        self.router_hidden_dim = router_hidden_dim
        self.router_temperature = router_temperature
        self.soft_temperature = soft_temperature
        self.text_recon_loss_weight = text_recon_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_loss_weight = contrastive_loss_weight

        # Store frame-level classification loss parameters
        self.num_frames = num_frames
        self.frame_temperature = frame_temperature

        # Store encoder/decoder types for forward pass branching
        self.encoder_type = encoder
        self.decoder_type = decoder

        # Instantiate encoder based on encoder parameter
        if encoder == "VideoEncoder":
            self.encoder = VideoEncoder(
                input_dim=in_dim,
                num_patches=num_patches,
                width=encoder_width,
                num_layers=encoder_layers,
                num_heads=encoder_heads,
                num_latent_tokens=num_latent_tokens,
                token_size=e_dim
            )
        elif encoder == "VideoLatentEncoder":
            self.encoder = VideoLatentEncoder(
                input_dim=in_dim,
                num_patches=num_patches,
                width=encoder_width,
                num_layers=encoder_layers,
                num_heads=encoder_heads,
                num_latent_tokens=num_latent_tokens,
                token_size=e_dim,
                version="1.1"  # Default to orthogonal initialization
            )
        elif encoder == "VideoSlotEncoder":
            self.encoder = VideoSlotEncoder(
                input_dim=in_dim,
                num_patches=num_patches,
                width=encoder_width,
                num_iterations=num_iterations,
                num_latent_tokens=num_latent_tokens
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder}.")

        # Project encoder output to quantization dimension if needed
        if encoder_width != e_dim:
            self.pre_quant_proj = nn.Linear(encoder_width, e_dim)
        else:
            self.pre_quant_proj = nn.Identity()

        # ResidualVectorQuantizer for quantizing latent tokens
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init=self.kmeans_init,
                                          kmeans_iters=self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear,
                                          beta=self.beta,
                                          ema_update=self.use_ema,
                                          decay=self.ema_decay)

        # Instantiate decoder based on decoder parameter
        if decoder == "VideoDecoder":
            self.decoder = VideoDecoder(
                output_dim=in_dim,
                num_patches=num_patches,
                width=encoder_width,
                num_layers=encoder_layers,
                num_heads=encoder_heads,
                num_latent_tokens=num_latent_tokens,
                token_size=e_dim
            )
        elif decoder == "VideoSlotMLPDecoder":
            self.decoder = VideoSlotMLPDecoder(
                output_dim=in_dim,
                num_patches=num_patches,
                width=encoder_width,
                num_layers=3,
                num_latent_tokens=num_latent_tokens,
                token_size=e_dim,
                decoder_input_dim=encoder_width
            )
        elif decoder == "VideoLatentDecoder":
            self.decoder = VideoLatentDecoder(
                output_dim=in_dim,
                num_patches=num_patches,
                width=encoder_width,
                num_layers=3,
                num_heads=encoder_heads,
                num_latent_tokens=num_latent_tokens,
                token_size=e_dim
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder}.")

        # Optional TextDecoder and Router for caption reconstruction
        if self.use_text_decoder:
            from .blocks import TextDecoder, LatentTokenRouter

            # LatentTokenRouter for selecting relevant latent tokens 
            self.latent_router = LatentTokenRouter(
                text_dim=text_dim,
                num_latent_tokens=num_latent_tokens,
                temperature=router_temperature,
                soft_temperature=soft_temperature
            )

            # TextDecoder for reconstructing text from selected quantized latent tokens
            self.text_decoder = TextDecoder(
                input_dim=e_dim,                      # Use e_dim to match q_video_emb dimension
                text_dim=text_dim,
                hidden_layers=text_decoder_layers,
                dropout_prob=dropout_prob,
                bn=bn
            )

    def get_config(self):
        """
        Return model configuration for checkpointing.
        """
        return self.config.copy()

    @classmethod
    def from_config(cls, config, **override_params):
        """
        Reconstruct VideoRQVAE from configuration dict.
        """
        merged_config = {**config, **override_params}

        # Filter out metadata fields that aren't model parameters
        metadata_keys = {'feature_extractor'}  # Add other metadata keys here as needed
        filtered_config = {k: v for k, v in merged_config.items() if k not in metadata_keys}

        return cls(**filtered_config)

    def _l2_normalize_patches(self, video_patches): 
        """Apply L2 normalization to video patches for stable training""" 
        eps = 1e-12 
        return F.normalize(video_patches, p=2, dim=-1, eps=eps)

    def forward(self, video_patches, text_embs=None, use_sk=True):
        """
        Forward pass for InternVL video patch sequences with optional multi-text reconstruction.

        Args:
            video_patches: [batch_size, num_patches, in_dim] - InternVL video patch features
            text_embs: [batch_size, num_texts, text_dim] - Multiple text embeddings per video (optional)
            use_sk: Whether to use Sinkhorn for quantization

        Returns:
            reconstructed: [batch_size, num_patches, in_dim] - Reconstructed video patches
            rq_loss: Quantization loss
            indices: Quantization indices
            x_encoded: Encoded features before quantization
            reconstructed_texts: List[[batch_size, text_dim]] - Reconstructed texts (if text_embs provided and text decoder enabled)
            selection_weights_all: List[[batch_size, num_latent_tokens]] - Router selection weights for each text
        """
        video_patches = self._l2_normalize_patches(video_patches)
        if text_embs is not None:
            text_embs = self._l2_normalize_patches(text_embs)

        # Forward pass through encoder - encoder type specific
        if self.encoder_type == "VideoSlotEncoder":
            # VideoSlotEncoder initializes slots internally (Gaussian initialization)
            x_encoded = self.encoder(video_patches)
        else:
            # VideoEncoder and VideoLatentEncoder use learnable latent tokens
            latent_tokens = self.encoder.learnable_latent_tokens
            x_encoded = self.encoder(video_patches, latent_tokens)

        # Project to quantization dimension if needed: [batch, num_latent_tokens, width] -> [batch, num_latent_tokens, e_dim]
        x_encoded = self.pre_quant_proj(x_encoded)

        # Quantize latent tokens: [batch, num_latent_tokens, e_dim]
        q_video_emb, rq_loss, indices, distances = self.rq(x_encoded, use_sk=use_sk)

        # VideoDecoder reconstructs video patches from quantized latent tokens
        # x_q: [batch, num_latent_tokens, e_dim] -> reconstructed: [batch, num_patches, in_dim]
        reconstructed = self.decoder(q_video_emb)

        # Multi-text reconstruction using quantized embeddings
        if text_embs is not None and self.use_text_decoder:
            # text_embs: [batch_size, num_texts, text_dim]
            batch_size, num_texts, text_dim = text_embs.shape

            # Batched processing: reshape for parallel computation
            # Flatten batch and text dimensions: [batch_size, num_texts, text_dim] -> [batch_size * num_texts, text_dim]
            text_flat = text_embs.view(batch_size * num_texts, text_dim)

            # Expand video embeddings to match flattened text batch
            # [batch_size, num_latent_tokens, e_dim] -> [batch_size * num_texts, num_latent_tokens, e_dim]
            q_video_expanded = q_video_emb.unsqueeze(1).expand(-1, num_texts, -1, -1).contiguous()
            q_video_flat = q_video_expanded.view(batch_size * num_texts, self.num_latent_tokens, -1)

            # Single batched router call for all texts
            selected_q_emb_flat, selection_weights_flat, selected_token_idx_flat = self.latent_router(text_flat, q_video_flat, return_token_idx=True)

            # Single batched decoder call for all texts
            reconstructed_text_flat = self.text_decoder(selected_q_emb_flat)  # [batch_size * num_texts, text_dim]

            # Reshape back to structured format: [batch_size * num_texts, ...] -> [batch_size, num_texts, ...]
            reconstructed_text_tensor = reconstructed_text_flat.view(batch_size, num_texts, text_dim)
            selection_weights_tensor = selection_weights_flat.view(batch_size, num_texts, -1)
            selected_token_idx_tensor = selected_token_idx_flat.view(batch_size, num_texts)

            # Convert to list format for loss computation
            reconstructed_texts = [reconstructed_text_tensor[:, i] for i in range(num_texts)]
            selection_weights_all = [selection_weights_tensor[:, i] for i in range(num_texts)]
            selected_token_idx_all = [selected_token_idx_tensor[:, i] for i in range(num_texts)]

            return reconstructed, rq_loss, indices, x_encoded, reconstructed_texts, selection_weights_all, selected_token_idx_all
        else:
            return reconstructed, rq_loss, indices, x_encoded

    @torch.no_grad()
    def get_indices(self, video_patches, use_sk=False, text_embs=None, return_semantic_selections=False, return_quantized_features=False):
        """
        Get quantization indices with unified support for collision metrics and semantic ID tracking.

        Args:
            video_patches: [batch_size, num_patches, in_dim] - InternVL video patch features
            use_sk: Whether to use Sinkhorn for quantization
            text_embs: Optional[torch.Tensor] - Text embeddings for guided selection
                      [batch_size, text_dim] for single text per video
                      [batch_size, num_texts, text_dim] for multi-text semantic tracking
            return_semantic_selections: bool - Whether to return per-text token selections for frequency analysis
            return_quantized_features: bool - Whether to return quantized embeddings for sID feature extraction

        Returns:
            indices: Quantization indices [batch_size, num_tokens, num_layers] for collision metrics
            quantized_features: Optional[torch.Tensor] - Quantized embeddings [batch_size, num_tokens, e_dim]
                               (only when return_quantized_features=True)
            semantic_selections: Optional[torch.Tensor] - Selected token indices per text [batch_size, num_texts]
                                (only when return_semantic_selections=True and text_embs provided)
        """
        video_patches = self._l2_normalize_patches(video_patches)

        # Single encoder/quantizer pass for efficiency - encoder type specific
        if self.encoder_type == "VideoSlotEncoder":
            # VideoSlotEncoder initializes slots internally
            x_e = self.encoder(video_patches)
        else:
            # VideoEncoder and VideoLatentEncoder use learnable latent tokens
            latent_tokens = self.encoder.learnable_latent_tokens
            x_e = self.encoder(video_patches, latent_tokens)
        x_e = self.pre_quant_proj(x_e)
        q_video_emb, _, all_indices, distances = self.rq(x_e, use_sk=use_sk)

        # Return quantized features when requested (for sID feature extraction)
        if return_quantized_features:
            return all_indices, q_video_emb

        # Standard case: return all token indices for collision metrics (backward compatible)
        if text_embs is None or not return_semantic_selections:
            if text_embs is not None and text_embs.dim() == 2:
                # Legacy single-text mode: [batch_size, text_dim]
                return self._get_single_text_indices(text_embs, q_video_emb, all_indices, distances)
            return all_indices, distances

        # Enhanced case: multi-text semantic ID tracking
        if (hasattr(self, 'latent_router') and self.use_text_decoder and
            self.num_latent_tokens > 1 and text_embs.dim() == 3):

            semantic_selections = self._get_multi_text_selections(text_embs, q_video_emb, all_indices)
            return all_indices, semantic_selections

        # Fallback: return standard indices only
        return all_indices, None

    def _get_single_text_indices(self, text_emb, q_video_emb, all_indices, distances):
        """Handle legacy single-text guided selection."""
        if (hasattr(self, 'latent_router') and self.use_text_decoder and self.num_latent_tokens > 1):
            _, _, selected_token_idx = self.latent_router(text_emb, q_video_emb, return_token_idx=True)
            batch_indices = torch.arange(text_emb.shape[0], device=text_emb.device)
            selected_indices = all_indices[batch_indices, selected_token_idx]
            return selected_indices, distances
        return all_indices[:, 0], distances

    def _get_multi_text_selections(self, text_embs, q_video_emb, all_indices):
        """
        Batch process multi-text semantic ID selections efficiently.

        Args:
            text_embs: [batch_size, num_texts, text_dim]
            q_video_emb: [batch_size, num_latent_tokens, e_dim]
            all_indices: [batch_size, num_latent_tokens, num_layers]

        Returns:
            semantic_selections: [batch_size, num_texts] - Selected token indices per video/text pair
        """
        batch_size, num_texts, text_dim = text_embs.shape

        # Reshape for batch processing: [batch_size * num_texts, text_dim]
        text_flat = text_embs.view(batch_size * num_texts, text_dim)

        # Expand video embeddings to match: [batch_size * num_texts, num_latent_tokens, e_dim]
        q_video_expanded = q_video_emb.unsqueeze(1).expand(-1, num_texts, -1, -1).contiguous()
        q_video_flat = q_video_expanded.view(batch_size * num_texts, self.num_latent_tokens, -1)

        # Batch router selection: [batch_size * num_texts]
        _, _, selected_flat = self.latent_router(text_flat, q_video_flat, return_token_idx=True)

        # Reshape back to per-video format: [batch_size, num_texts]
        selected_token_indices = selected_flat.view(batch_size, num_texts)

        return selected_token_indices

    def compute_loss(self, recon_video_features, encoder_out, quant_loss, video_features,
                     text_embs=None, reconstructed_texts=None, selection_weights_all=None):
        """
        Compute comprehensive loss including video reconstruction, frame classification, multi-text reconstruction, and router diversity.

        Args:
            recon_video_features: [batch_size, num_patches, in_dim] - Reconstructed video patches
            encoder_out: Encoded features before quantization (unused, kept for compatibility)
            quant_loss: Quantization loss from RQ-VAE
            video_features: [batch_size, num_patches, in_dim] - Original video patches
            text_embs: [batch_size, num_texts, text_dim] - Original text embeddings (optional)
            reconstructed_texts: List of [batch_size, text_dim] - Reconstructed text embeddings (optional)
            selection_weights_all: List of [batch_size, num_latent_tokens] - Router weights for each text (optional)

        Returns:
            loss_total: Total weighted loss
            loss_recon: Video reconstruction loss (frame classification when loss_type='cls')
            loss_multi_text: Multi-text reconstruction loss (if applicable)
            loss_diversity: Router diversity loss (if applicable)
        """

        # Loss 1: Video Reconstruction Loss (Flexible Weighted Combination)
        recon_video_features = self._l2_normalize_patches(recon_video_features)
        video_features = self._l2_normalize_patches(video_features)
        # Compute weighted sum based on vid_loss_weight [mse, l1, cosine, cls, p2p_infonce]
        loss_recon = 0.0
        # MSE Loss (weight index 0)
        if self.vid_loss_weight[0] > 0:
            loss_mse = F.mse_loss(recon_video_features, video_features, reduction='sum') / (recon_video_features.shape[0] * recon_video_features.shape[1])
            loss_recon += self.vid_loss_weight[0] * loss_mse
        # L1 Loss (weight index 1)
        if self.vid_loss_weight[1] > 0:
            loss_l1 = F.l1_loss(recon_video_features, video_features, reduction='mean')
            loss_recon += self.vid_loss_weight[1] * loss_l1
        # Cosine Loss (weight index 2)
        if self.vid_loss_weight[2] > 0:
            loss_cosine = (1 - F.cosine_similarity(recon_video_features, video_features, dim=-1)).mean()
            loss_recon += self.vid_loss_weight[2] * loss_cosine
        # Frame Classification Loss (weight index 3)
        if self.vid_loss_weight[3] > 0:
            loss_cls = self._compute_frame_classification_loss(recon_video_features, video_features)
            loss_recon += self.vid_loss_weight[3] * loss_cls
        # Patch-to-Patch InfoNCE Loss (weight index 4)
        if self.vid_loss_weight[4] > 0:
            loss_p2p_infonce = self._compute_patch_to_patch_infonce_loss(recon_video_features, video_features)
            loss_recon += self.vid_loss_weight[4] * loss_p2p_infonce

        # Loss 2: Multi-Text Reconstruction Loss
        loss_multi_text = torch.tensor(0.0, device=loss_recon.device)
        if text_embs is not None and reconstructed_texts is not None:
            loss_multi_text = self._compute_multi_text_reconstruction_loss(text_embs, reconstructed_texts)

        # Loss 3: Router Diversity Loss (conditional on num_latent_tokens > 1)
        loss_diversity = torch.tensor(0.0, device=loss_recon.device)
        if (selection_weights_all is not None and
            self.num_latent_tokens > 1 and
            hasattr(self, 'diversity_loss_weight')):
            loss_diversity = self._compute_router_diversity_loss(selection_weights_all)

        # Total weighted loss
        loss_total = (loss_recon +
                     self.quant_loss_weight * quant_loss)

        if text_embs is not None and reconstructed_texts is not None:
            loss_multi_text = self.text_recon_loss_weight * loss_multi_text
            loss_total += loss_multi_text

        if self.num_latent_tokens > 1 and hasattr(self, 'diversity_loss_weight'):
            loss_diversity = self.diversity_loss_weight * loss_diversity
            loss_total += loss_diversity

        return loss_total, loss_recon, loss_multi_text, loss_diversity

    def _compute_frame_prototypes(self, video_features):
        """
        Compute normalized frame prototypes by averaging patches within each frame.

        Args:
            video_features: [batch_size, num_patches, in_dim] - Original video patches

        Returns:
            frame_prototypes: [batch_size, num_frames, in_dim] - Normalized frame prototypes
        """
        batch_size, num_patches, in_dim = video_features.shape

        if self.num_frames is None or self.num_frames <= 0:
            raise ValueError("num_frames must be set to use frame-level classification loss")

        if num_patches % self.num_frames != 0:
            raise ValueError(f"num_patches ({num_patches}) must be divisible by num_frames ({self.num_frames})")

        patches_per_frame = num_patches // self.num_frames

        # Reshape patches into frames: [batch_size, num_frames, patches_per_frame, in_dim]
        reshaped = video_features.view(batch_size, self.num_frames, patches_per_frame, in_dim)

        # Average patches within each frame: [batch_size, num_frames, in_dim]
        frame_prototypes = reshaped.mean(dim=2)

        # Normalize frame prototypes (stop gradient for ground truth)
        eps = 1e-12
        frame_prototypes = F.normalize(frame_prototypes.detach(), p=2, dim=-1, eps=eps)

        return frame_prototypes

    def _compute_frame_classification_loss(self, recon_video_features, video_features):
        """
        Compute frame-level classification loss treating each frame as a class.

        Every reconstructed patch should match its own frame prototype and be far from others.
        Uses temperature-scaled cross-entropy loss.

        Args:
            recon_video_features: [batch_size, num_patches, in_dim] - Reconstructed video patches
            video_features: [batch_size, num_patches, in_dim] - Original video patches

        Returns:
            Frame-level classification loss
        """
        if self.num_frames is None or self.num_frames <= 1:
            return torch.tensor(0.0, device=recon_video_features.device)

        batch_size, num_patches, in_dim = recon_video_features.shape
        patches_per_frame = num_patches // self.num_frames

        # Compute ground-truth frame prototypes
        frame_prototypes = self._compute_frame_prototypes(video_features)  # [batch_size, num_frames, in_dim]

        # Compute similarity between each reconstructed patch and all frame prototypes
        # [batch_size, num_patches, in_dim] × [batch_size, in_dim, num_frames] -> [batch_size, num_patches, num_frames]
        logits = torch.matmul(recon_video_features, frame_prototypes.transpose(1, 2)) / self.frame_temperature

        # Create labels: each patch belongs to its corresponding frame
        # Patches 0 to patches_per_frame-1 belong to frame 0, etc.
        labels = torch.arange(num_patches, device=recon_video_features.device) // patches_per_frame
        labels = labels.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_patches]

        # Reshape for cross-entropy: [batch_size * num_patches, num_frames]
        logits_flat = logits.view(-1, self.num_frames)
        labels_flat = labels.reshape(-1)

        # Compute cross-entropy loss
        frame_loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')

        return frame_loss

    def _compute_patch_to_patch_infonce_loss(self, recon_video_features, video_features):
        """
        Compute patch-to-patch InfoNCE loss within each frame.

        For each reconstructed patch in a frame:
        - Positive: corresponding GT patch at the same index within the frame
        - Negatives: all other GT patches within the same frame

        Uses frame structure defined by self.num_frames.

        Args:
            recon_video_features: [batch_size, num_patches, in_dim] - Reconstructed video patches
            video_features: [batch_size, num_patches, in_dim] - Ground truth video patches

        Returns:
            Patch-to-patch InfoNCE loss averaged over all patches
        """
        if self.num_frames is None or self.num_frames <= 0:
            raise ValueError("num_frames must be set to use p2p_infonce loss")

        batch_size, num_patches, in_dim = recon_video_features.shape

        if num_patches % self.num_frames != 0:
            raise ValueError(f"num_patches ({num_patches}) must be divisible by num_frames ({self.num_frames})")

        patches_per_frame = num_patches // self.num_frames

        # Reshape for batched matmul: [batch_size * num_frames, patches_per_frame, in_dim]
        recon_frames = recon_video_features.view(batch_size * self.num_frames, patches_per_frame, in_dim)
        video_frames = video_features.view(batch_size * self.num_frames, patches_per_frame, in_dim)

        # Compute similarity matrix within each frame: [batch_size * num_frames, patches_per_frame, patches_per_frame]
        # similarity[frame_idx, i, j] = similarity between reconstructed patch i and GT patch j in the same frame
        similarity = torch.matmul(recon_frames, video_frames.transpose(1, 2)) / self.contrastive_temperature

        # Flatten for cross-entropy: [batch_size * num_frames * patches_per_frame, patches_per_frame]
        # Each row: one reconstructed patch compared against all GT patches in its frame
        similarity_flat = similarity.view(-1, patches_per_frame)

        # Labels: each reconstructed patch should match GT patch at the same index (diagonal)
        # For each frame, labels are [0, 1, 2, ..., patches_per_frame-1]
        labels = torch.arange(patches_per_frame, device=similarity.device).repeat(batch_size * self.num_frames)

        # Standard InfoNCE cross-entropy loss
        loss = F.cross_entropy(similarity_flat, labels, reduction='mean')

        return loss

    def _compute_multi_text_reconstruction_loss(self, text_embs, reconstructed_texts):
        """
        Compute reconstruction loss with enhanced InfoNCE for multi-positive contrastive learning.

        For contrastive loss: treats all texts from the same video as positives,
        and texts from different videos as negatives within the batch.

        Args:
            text_embs: [batch_size, num_texts, text_dim] - Original text embeddings
            reconstructed_texts: List of [batch_size, text_dim] - Reconstructed texts

        Returns:
            Average text reconstruction loss across all texts
        """
        _, num_texts, _ = text_embs.shape

        if self.text_loss_type == 'mse':
            # Batched MSE approach - stack list back to tensor for efficient computation
            # reconstructed_texts: List[[batch_size, text_dim]] -> [batch_size, num_texts, text_dim]
            reconstructed_tensor = torch.stack(reconstructed_texts, dim=1)
            # text_embs: [batch_size, num_texts, text_dim]
            # Compute MSE over all dimensions and average
            text_loss = F.mse_loss(reconstructed_tensor, text_embs, reduction='mean')
            return text_loss

        elif self.text_loss_type == 'contrastive':
            # Enhanced InfoNCE with ALL captions from other videos as negatives
            return self._compute_infonce_loss(text_embs, reconstructed_texts)
        else:
            raise ValueError(f'Invalid text_loss_type: {self.text_loss_type}')

    def _compute_infonce_loss(self, text_embs, reconstructed_texts):
        """
        Compute InfoNCE loss where each reconstructed text competes against ALL captions from other videos.

        For each reconstructed text:
        - Positive: corresponding GT caption (same video, same caption index)
        - Negatives: ALL captions from other videos (comprehensive negative sampling)

        Args:
            text_embs: [batch_size, num_texts, text_dim] - Original text embeddings
            reconstructed_texts: List of [batch_size, text_dim] - Reconstructed texts

        Returns:
            InfoNCE loss with comprehensive negative sampling
        """
        batch_size, num_texts, text_dim = text_embs.shape

        # Flatten all texts using consistent ordering: [batch_size * num_texts, text_dim]
        # Order: [video0_text0, video0_text1, ..., video1_text0, video1_text1, ...]
        all_targets = text_embs.view(-1, text_dim).detach()

        # Reshape reconstructed_texts to match the same ordering
        all_reconstructed = torch.stack(reconstructed_texts, dim=1).view(-1, text_dim)

        # Normalize for cosine similarity
        eps = 1e-12
        all_reconstructed_norm = F.normalize(all_reconstructed, p=2, dim=1, eps=eps)
        all_targets_norm = F.normalize(all_targets, p=2, dim=1, eps=eps)

        # Compute similarity matrix: [batch_size * num_texts, batch_size * num_texts]
        similarity_matrix = torch.matmul(all_reconstructed_norm, all_targets_norm.t()) / self.contrastive_temperature

        # Vectorized InfoNCE computation - O(1) instead of O(B*T) loop
        device = similarity_matrix.device

        # Create video and text index mappings for broadcasting
        video_ids = torch.arange(batch_size).repeat_interleave(num_texts).to(device)
        text_indices = torch.arange(num_texts).repeat(batch_size).to(device)

        # Create masks using broadcasting: [B*T, 1] × [1, B*T] -> [B*T, B*T]
        video_ids_anchors = video_ids.unsqueeze(1)     # [B*T, 1]
        video_ids_targets = video_ids.unsqueeze(0)      # [1, B*T]
        text_indices_anchors = text_indices.unsqueeze(1)  # [B*T, 1]
        text_indices_targets = text_indices.unsqueeze(0)   # [1, B*T]

        # Positive mask: same video AND same text index (should be diagonal)
        positive_mask = ((video_ids_anchors == video_ids_targets) &
                        (text_indices_anchors == text_indices_targets)).float()

        # Negative mask: different video (ignore text index for negatives)
        negative_mask = (video_ids_anchors != video_ids_targets).float()

        # Combined valid targets mask: positives + negatives
        valid_mask = positive_mask + negative_mask  # [B*T, B*T]

        # Apply mask to similarities (set invalid targets to -inf)
        masked_similarities = similarity_matrix.masked_fill(valid_mask == 0, float('-inf'))

        # Vectorized InfoNCE: loss_i = -sim[i,i] + logsumexp(valid_sims[i])
        # Positive similarities are always on the diagonal
        positive_sims = torch.diag(similarity_matrix)  # [B*T]

        # Compute logsumexp over valid targets for each anchor
        logsumexp_terms = torch.logsumexp(masked_similarities, dim=1)  # [B*T]

        # InfoNCE loss for each anchor: -log(exp(pos) / exp(pos + negatives))
        individual_losses = -positive_sims + logsumexp_terms  # [B*T]

        # Return mean loss across all anchors
        return individual_losses.mean()

    def _compute_router_diversity_loss(self, selection_weights_all):
        """
        Compute router diversity loss to encourage different token selections for different texts.
        Only activated when num_latent_tokens > 1.

        Args:
            selection_weights_all: List of [batch_size, num_latent_tokens] - Router weights for each text

        Returns:
            Router diversity loss encouraging different token selections
        """
        if self.num_latent_tokens <= 1:
            # No diversity possible with single token
            return torch.tensor(0.0, device=selection_weights_all[0].device)

        # Stack all selection weights: [batch_size, num_texts, num_latent_tokens]
        all_weights = torch.stack(selection_weights_all, dim=1)
        batch_size, num_texts, _ = all_weights.shape

        diversity_losses = []

        for b in range(batch_size):
            batch_weights = all_weights[b]  # [num_texts, num_latent_tokens]

            # Compute pairwise cosine similarities between different text selections
            # Normalize weights for cosine similarity
            normalized_weights = F.normalize(batch_weights, p=2, dim=-1)  # [num_texts, num_latent_tokens]

            # Compute similarity matrix: [num_texts, num_texts]
            similarities = torch.matmul(normalized_weights, normalized_weights.t())

            # Extract off-diagonal elements (exclude self-similarity)
            mask = ~torch.eye(num_texts, dtype=torch.bool, device=similarities.device)
            off_diagonal_sims = similarities[mask]

            # Penalize high similarities - we want diversity (low similarity)
            diversity_loss = off_diagonal_sims.mean()
            diversity_losses.append(diversity_loss)

        return torch.stack(diversity_losses).mean()


    def compute_loss_simple(self, recon_video_features, quant_loss, video_features=None):
        """Backward compatibility method for simple loss computation without text loss"""
        # Compute weighted sum based on vid_loss_weight [mse, l1, cosine, cls, p2p_infonce]
        loss_recon = 0.0
        # MSE Loss (weight index 0)
        if self.vid_loss_weight[0] > 0:
            loss_mse = F.mse_loss(recon_video_features, video_features, reduction='mean')
            loss_recon += self.vid_loss_weight[0] * loss_mse
        # L1 Loss (weight index 1)
        if self.vid_loss_weight[1] > 0:
            loss_l1 = F.l1_loss(recon_video_features, video_features, reduction='mean')
            loss_recon += self.vid_loss_weight[1] * loss_l1
        # Cosine Loss (weight index 2)
        if self.vid_loss_weight[2] > 0:
            loss_cosine = (1 - F.cosine_similarity(recon_video_features, video_features, dim=-1)).mean()
            loss_recon += self.vid_loss_weight[2] * loss_cosine
        # Frame Classification Loss (weight index 3)
        if self.vid_loss_weight[3] > 0:
            loss_cls = self._compute_frame_classification_loss(recon_video_features, video_features)
            loss_recon += self.vid_loss_weight[3] * loss_cls
        # Patch-to-Patch InfoNCE Loss (weight index 4)
        if self.vid_loss_weight[4] > 0:
            loss_p2p_infonce = self._compute_patch_to_patch_infonce_loss(recon_video_features, video_features)
            loss_recon += self.vid_loss_weight[4] * loss_p2p_infonce

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon


# class VideoRQVAE_V2(nn.Module):
#     def __init__(self,
#                  in_dim=512,                  
#                  num_patches=1,              
#                  num_latent_tokens=4,
#                  encoder_width=512,
#                  encoder_layers=1,
#                  encoder_heads=8,
#                  num_emb_list=[256,256,256,256],
#                  e_dim=512,
#                  text_loss_type="mse",
#                  text_loss_pos="after",
#                  quant_loss_weight=1.0,
#                  kmeans_init=True,
#                  kmeans_iters=100,
#                  sk_epsilons=None,
#                  sk_iters=100,
#                  use_linear=0,
#                  dropout_prob=0.0,            # Added: for backward compatibility
#                  bn=False,                    # Added: for backward compatibility
#                  # EMA parameters
#                  use_ema=False,               # Use EMA for codebook updates instead of gradients
#                  ema_decay=0.99,              # Decay rate for EMA updates
#                  # TextDecoder parameters
#                  use_text_decoder=False,      # Whether to include text decoder
#                  text_dim=512,               # Target text embedding dimension
#                  text_decoder_layers=[1536, 3072],    # Hidden layers for text decoder
#                  # Router parameters
#                  router_hidden_dim=512,       # Hidden dimension for router
#                  router_temperature=1.5,      # Temperature for router similarity computation
#                  soft_temperature=1.0,        # Temperature for soft mixture control
#                  # Text reconstruction loss parameter
#                  text_recon_loss_weight=1.0,  # Weight for text reconstruction loss
#                  # Router diversity loss parameter (optional, disabled if num_latent_tokens=1)
#                  diversity_loss_weight=0.1,   # Weight for router diversity loss
#                  # Contrastive loss parameters
#                  contrastive_temperature=0.07,  # Temperature for InfoNCE contrastive loss
#                  contrastive_loss_weight=1.0,  # Weight for contrastive loss
#                  beta=0.55,  # Beta for vq loss
#                  version=1.0,  # Keep for experiment naming only (not used for selection)
#                  num_iterations=3,  # Number of slot attention iterations (for VideoSlotEncoder)
#                  # Encoder/decoder selection
#                  encoder="VideoEncoder",
#                  decoder="VideoDecoder",
#                  # Frame-level classification loss parameters
#                  num_frames=8,  # Number of frames per video (required for loss_type='cls')
#                  frame_temperature=0.1,  # Temperature for frame classification softmax
#                  # Flexible video reconstruction loss weighting
#                  vid_loss_weight=None  # Array [mse, l1, cosine, cls, p2p_infonce] weights; None = auto-convert from loss_type
#         ):
#         super(VideoRQVAE_V2, self).__init__()

#         # Store complete configuration BEFORE importing classes (captures only __init__ params)
#         self.config = locals().copy()
#         self.config.pop('self')
#         self.config.pop('__class__', None)

#         # Import encoder and decoder classes from dedicated files
#         from .encoder import VideoLatentEncoder_V2
#         from .decoder import VideoLatentDecoder_V2

#         self.in_dim = in_dim                    # Input dimension per patch (from InternVL)
#         self.input_dim = in_dim                 # For backward compatibility
#         self.num_patches = num_patches
#         self.num_latent_tokens = num_latent_tokens
#         self.encoder_width = encoder_width
#         self.num_emb_list = num_emb_list
#         self.e_dim = e_dim
#         self.vid_loss_weight = vid_loss_weight  
#         self.text_loss_type = text_loss_type
#         self.text_loss_pos = text_loss_pos
#         self.quant_loss_weight = quant_loss_weight
#         self.kmeans_init = kmeans_init
#         self.kmeans_iters = kmeans_iters
#         self.sk_epsilons = sk_epsilons
#         self.sk_iters = sk_iters
#         self.beta = beta
#         self.num_iterations = num_iterations
#         self.version = version  

#         # Store backward compatibility parameters
#         self.dropout_prob = dropout_prob
#         self.bn = bn

#         # Store EMA parameters
#         self.use_ema = use_ema
#         self.ema_decay = ema_decay

#         # Store text decoder parameters
#         self.use_text_decoder = use_text_decoder
#         self.text_dim = text_dim
#         self.text_decoder_layers = text_decoder_layers
#         self.router_hidden_dim = router_hidden_dim
#         self.router_temperature = router_temperature
#         self.soft_temperature = soft_temperature
#         self.text_recon_loss_weight = text_recon_loss_weight
#         self.diversity_loss_weight = diversity_loss_weight
#         self.contrastive_temperature = contrastive_temperature
#         self.contrastive_loss_weight = contrastive_loss_weight

#         # Store frame-level classification loss parameters
#         self.num_frames = num_frames
#         self.frame_temperature = frame_temperature

#         self.encoder = VideoLatentEncoder_V2(
#             input_dim=in_dim,
#             width=encoder_width,
#             num_layers=encoder_layers,
#             num_heads=encoder_heads,
#             num_latent_tokens=num_latent_tokens,
#             token_size=e_dim
#         )

#         # Project encoder output to quantization dimension if needed
#         if encoder_width != e_dim:
#             self.pre_quant_proj = nn.Linear(encoder_width, e_dim)
#         else:
#             self.pre_quant_proj = nn.Identity()

#         # ResidualVectorQuantizer for quantizing latent tokens
#         self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
#                                           kmeans_init=self.kmeans_init,
#                                           kmeans_iters=self.kmeans_iters,
#                                           sk_epsilons=self.sk_epsilons,
#                                           sk_iters=self.sk_iters,
#                                           use_linear=use_linear,
#                                           beta=self.beta,
#                                           ema_update=self.use_ema,
#                                           decay=self.ema_decay)

#         self.decoder = VideoLatentDecoder_V2(
#             output_dim=in_dim,
#             width=encoder_width,
#             num_layers=encoder_layers,
#             num_heads=encoder_heads,
#             num_latent_tokens=num_latent_tokens,
#             token_size=e_dim
#         )


#     def get_config(self):
#         """
#         Return model configuration for checkpointing.
#         """
#         return self.config.copy()

#     @classmethod
#     def from_config(cls, config, **override_params):
#         """
#         Reconstruct VideoRQVAE_V2 from configuration dict.
#         """
#         merged_config = {**config, **override_params}

#         # Filter out metadata fields that aren't model parameters
#         metadata_keys = {'feature_extractor'}  # Add other metadata keys here as needed
#         filtered_config = {k: v for k, v in merged_config.items() if k not in metadata_keys}

#         return cls(**filtered_config)

#     def forward(self, video_patches, text_embs=None, use_sk=True):
#         """
#         Forward pass for InternVideo2 pooled video features.

#         Args:
#             video_patches: [batch_size, in_dim] - Single pooled video feature per video
#             text_embs: Optional [batch_size, num_texts, text_dim] for contrastive learning
#             use_sk: Whether to use Sinkhorn for quantization

#         Returns:
#             reconstructed: [batch_size, in_dim] - Reconstructed video feature
#             rq_loss: Quantization loss
#             indices: Quantization indices
#             x_encoded: Encoded features before quantization
#             q_video_emb: [batch_size, num_latent_tokens, e_dim] - Quantized video embeddings for contrastive loss
#         """
#         # VideoEncoder and VideoLatentEncoder use learnable latent tokens
#         latent_tokens = self.encoder.learnable_latent_tokens
#         x_encoded = self.encoder(video_patches, latent_tokens)

#         # Project to quantization dimension if needed: [batch, num_latent_tokens, width] -> [batch, num_latent_tokens, e_dim]
#         x_encoded = self.pre_quant_proj(x_encoded)

#         # Quantize latent tokens: [batch, num_latent_tokens, e_dim]
#         q_video_emb, rq_loss, indices, distances = self.rq(x_encoded, use_sk=use_sk)

#         # VideoDecoder reconstructs video feature from quantized latent tokens
#         # x_q: [batch, num_latent_tokens, e_dim] -> reconstructed: [batch, in_dim]
#         reconstructed = self.decoder(q_video_emb)

#         return reconstructed, rq_loss, indices, x_encoded, q_video_emb

#     def _get_multi_text_selections(self, text_embs, q_video_emb):
#         """
#         Select latent tokens for each video-text pair using direct cosine similarity.

#         Args:
#             text_embs: [batch_size, num_texts, text_dim] - Text embeddings for each video
#             q_video_emb: [batch_size, num_latent_tokens, e_dim] - Quantized video embeddings

#         Returns:
#             semantic_selections: [batch_size, num_texts] - Selected token indices per video/text pair
#         """
#         # Normalize embeddings for cosine similarity
#         text_norm = F.normalize(text_embs, p=2, dim=-1)      # [B, num_texts, text_dim]
#         video_norm = F.normalize(q_video_emb, p=2, dim=-1)   # [B, num_latent_tokens, e_dim]

#         # Compute cosine similarity: [B, num_texts, num_latent_tokens]
#         # For each text, compute similarity with all latent tokens
#         similarity_matrix = torch.einsum('bnd,bkd->bnk', text_norm, video_norm)

#         # Select best matching token for each text via argmax
#         # semantic_selections[b, n] = index of token with highest similarity to text n
#         semantic_selections = torch.argmax(similarity_matrix, dim=-1)  # [B, num_texts]

#         return semantic_selections

#     @torch.no_grad()
#     def get_indices(self, video_patches, use_sk=False, text_embs=None, return_semantic_selections=False):
#         """
#         Get quantization indices for InternVideo2 pooled features.

#         Args:
#             video_patches: [batch_size, in_dim] - Single pooled video feature per video
#             use_sk: Whether to use Sinkhorn for quantization
#             text_embs: Optional[torch.Tensor]
#                       [batch_size, num_texts, text_dim] for multi-text semantic tracking
#             return_semantic_selections: bool - Whether to return per-text token selections

#         Returns:
#             indices: Quantization indices [batch_size, num_tokens, num_layers]
#             semantic_selections or distances:
#                 - If return_semantic_selections=True and text_embs provided:
#                   semantic_selections [batch_size, num_texts] - selected token indices per text
#                 - Otherwise: distance values for compatibility
#         """
#         # Single encoder/quantizer pass
#         latent_tokens = self.encoder.learnable_latent_tokens
#         x_e = self.encoder(video_patches, latent_tokens)
#         x_e = self.pre_quant_proj(x_e)
#         q_video_emb, _, all_indices, distances = self.rq(x_e, use_sk=use_sk)

#         # Standard case: return all token indices for collision metrics
#         if text_embs is None or not return_semantic_selections:
#             return all_indices, distances

#         # Multi-text semantic tracking: compute which token is selected per video-text pair
#         if text_embs.dim() == 3:
#             semantic_selections = self._get_multi_text_selections(text_embs, q_video_emb)
#             return all_indices, semantic_selections

#         # Fallback for unexpected input
#         return all_indices, distances

#     def compute_contrastive_loss(self, q_video_emb, text_embs):
#         """
#         Compute hard-positive contrastive loss between video tokens and text embeddings.

#         Strategy:
#         - Step 1: Compute cosine similarity between all video tokens and all texts
#         - Step 2: For each video token, find the highest scoring text FROM THE SAME VIDEO as positive
#         - Step 3: All texts from OTHER VIDEOS serve as negatives
#         - Step 4: Bidirectional InfoNCE loss (video→text + text→video), averaged

#         Args:
#             q_video_emb: [batch_size, num_latent_tokens, e_dim] - Quantized video embeddings
#             text_embs: [batch_size, num_texts, text_dim] - Text embeddings for each video

#         Returns:
#             contrastive_loss: Scalar tensor - Symmetric hard-positive InfoNCE loss
#         """
#         batch_size, num_tokens, e_dim = q_video_emb.shape
#         _, num_texts, text_dim = text_embs.shape
#         device = q_video_emb.device

#         # ============ Step 1: Compute Full Similarity Matrix ============
#         # Reshape for contrastive learning: [B, N, D] -> [B*N, D]
#         video_tokens = q_video_emb.reshape(batch_size * num_tokens, e_dim)  # [B*num_tokens, e_dim]
#         text_tokens = text_embs.reshape(batch_size * num_texts, text_dim)    # [B*num_texts, text_dim]

#         # L2 normalize embeddings for cosine similarity
#         video_tokens_norm = F.normalize(video_tokens, dim=-1, eps=1e-12)
#         text_tokens_norm = F.normalize(text_tokens, dim=-1, eps=1e-12)

#         # Compute similarity matrix: [B*num_tokens, B*num_texts]
#         similarity_matrix = torch.matmul(video_tokens_norm, text_tokens_norm.T) / self.contrastive_temperature

#         # ============ Video-to-Text Direction ============
#         # Step 2: Find highest scoring text FROM SAME VIDEO for each video token

#         # Create video ID mapping: [0,0,0,0, 1,1,1,1, ..., B-1,B-1,B-1,B-1] for num_tokens=4
#         video_ids_v = torch.arange(batch_size, device=device).repeat_interleave(num_tokens)  # [B*num_tokens]
#         video_ids_t = torch.arange(batch_size, device=device).repeat_interleave(num_texts)   # [B*num_texts]

#         # Create same-video mask: [B*num_tokens, B*num_texts]
#         # True where video token and text are from the same video
#         same_video_mask_v2t = video_ids_v.unsqueeze(1) == video_ids_t.unsqueeze(0)  # [B*num_tokens, B*num_texts]

#         # Step 3: Create masks for positive and negative pairs
#         # Mask out texts from other videos to find max within same video
#         masked_sim_v2t = similarity_matrix.clone()
#         masked_sim_v2t[~same_video_mask_v2t] = float('-inf')

#         # Find argmax (best matching text) for each video token within same video
#         positive_indices_v2t = torch.argmax(masked_sim_v2t, dim=1)  # [B*num_tokens]

#         # Create one-hot positive mask: each video token has exactly ONE positive
#         positive_mask_v2t = torch.zeros_like(similarity_matrix, dtype=torch.bool)
#         positive_mask_v2t[torch.arange(batch_size * num_tokens, device=device), positive_indices_v2t] = True

#         # Negative mask: all texts from OTHER videos
#         negative_mask_v2t = ~same_video_mask_v2t

#         # Valid mask: positives + negatives (exclude other texts from same video)
#         valid_mask_v2t = positive_mask_v2t | negative_mask_v2t

#         # Step 4: Compute InfoNCE loss (video→text)
#         masked_similarities_v2t = similarity_matrix.masked_fill(~valid_mask_v2t, float('-inf'))

#         # Get positive similarities
#         positive_sims_v2t = similarity_matrix[positive_mask_v2t].view(batch_size * num_tokens)  # [B*num_tokens]

#         # Compute logsumexp over valid targets
#         logsumexp_v2t = torch.logsumexp(masked_similarities_v2t, dim=1)  # [B*num_tokens]

#         # InfoNCE: -log(exp(pos) / exp(pos + negatives))
#         individual_losses_v2t = -positive_sims_v2t + logsumexp_v2t
#         loss_v2t = individual_losses_v2t.mean()

#         # ============ Text-to-Video Direction (Symmetric) ============
#         # Step 2: Find highest scoring video token FROM SAME VIDEO for each text

#         # Create same-video mask: [B*num_texts, B*num_tokens]
#         same_video_mask_t2v = video_ids_t.unsqueeze(1) == video_ids_v.unsqueeze(0)  # [B*num_texts, B*num_tokens]

#         # Transpose similarity matrix for text-to-video
#         similarity_matrix_t2v = similarity_matrix.T  # [B*num_texts, B*num_tokens]

#         # Step 3: Create masks for positive and negative pairs
#         masked_sim_t2v = similarity_matrix_t2v.clone()
#         masked_sim_t2v[~same_video_mask_t2v] = float('-inf')

#         # Find argmax (best matching video token) for each text within same video
#         positive_indices_t2v = torch.argmax(masked_sim_t2v, dim=1)  # [B*num_texts]

#         # Create one-hot positive mask
#         positive_mask_t2v = torch.zeros_like(similarity_matrix_t2v, dtype=torch.bool)
#         positive_mask_t2v[torch.arange(batch_size * num_texts, device=device), positive_indices_t2v] = True

#         # Negative mask: all video tokens from OTHER videos
#         negative_mask_t2v = ~same_video_mask_t2v

#         # Valid mask: positives + negatives
#         valid_mask_t2v = positive_mask_t2v | negative_mask_t2v

#         # Step 4: Compute InfoNCE loss (text→video)
#         masked_similarities_t2v = similarity_matrix_t2v.masked_fill(~valid_mask_t2v, float('-inf'))

#         # Get positive similarities
#         positive_sims_t2v = similarity_matrix_t2v[positive_mask_t2v].view(batch_size * num_texts)  # [B*num_texts]

#         # Compute logsumexp over valid targets
#         logsumexp_t2v = torch.logsumexp(masked_similarities_t2v, dim=1)  # [B*num_texts]

#         # InfoNCE: -log(exp(pos) / exp(pos + negatives))
#         individual_losses_t2v = -positive_sims_t2v + logsumexp_t2v
#         loss_t2v = individual_losses_t2v.mean()

#         # ============ Final Symmetric Loss ============
#         contrastive_loss = (loss_v2t + loss_t2v) / 2

#         return contrastive_loss

#     def compute_contrastive_loss_test(self, q_video_emb, text_embs):
#         """
#         Compute contrastive loss for test set where each video has exactly ONE query.

#         Strategy:
#         1. For each video, select the latent token with highest cosine similarity to its single text
#         2. Build [batch_size, batch_size] similarity matrix using selected tokens
#         3. Apply InfoNCE loss (diagonal = positives, off-diagonal = negatives)

#         Args:
#             q_video_emb: [batch_size, num_latent_tokens, e_dim] - Quantized video embeddings
#             text_embs: [batch_size, text_dim] or [batch_size, 1, text_dim] - Single text per video

#         Returns:
#             contrastive_loss: Scalar tensor - Test set contrastive loss
#         """
#         batch_size, num_tokens, e_dim = q_video_emb.shape
#         device = q_video_emb.device

#         # Step 1: Handle text_embs shape - squeeze if [B, 1, text_dim]
#         if text_embs.dim() == 3 and text_embs.shape[1] == 1:
#             text_embs = text_embs.squeeze(1)  # [B, 1, text_dim] -> [B, text_dim]

#         # Step 2: Normalize embeddings for cosine similarity
#         video_tokens_norm = F.normalize(q_video_emb, dim=-1, eps=1e-12)  # [B, num_tokens, e_dim]
#         text_norm = F.normalize(text_embs, dim=-1, eps=1e-12)  # [B, text_dim]

#         # Step 3: Compute similarity between each video's tokens and its corresponding text
#         # For each video, compute sim(all_tokens, single_text)
#         # Expand text for broadcasting: [B, text_dim] -> [B, 1, text_dim]
#         text_expanded = text_norm.unsqueeze(1)  # [B, 1, text_dim]

#         # Compute per-video token-text similarity: [B, num_tokens]
#         # For video i, compute similarity between its num_tokens and its single text
#         per_video_similarity = torch.sum(video_tokens_norm * text_expanded, dim=-1)  # [B, num_tokens]

#         # Step 4: Select token with highest similarity for each video
#         best_token_indices = torch.argmax(per_video_similarity, dim=1)  # [B]

#         # Gather selected tokens: [B, e_dim]
#         batch_indices = torch.arange(batch_size, device=device)
#         selected_video_emb = q_video_emb[batch_indices, best_token_indices]  # [B, e_dim]

#         # Step 5: Normalize selected embeddings
#         selected_video_norm = F.normalize(selected_video_emb, dim=-1, eps=1e-12)  # [B, e_dim]

#         # Step 6: Build [batch_size, batch_size] similarity matrix
#         # similarity[i, j] = similarity between video i's selected token and text j
#         similarity_matrix = torch.matmul(selected_video_norm, text_norm.T) / self.contrastive_temperature  # [B, B]

#         # Step 7: Bidirectional InfoNCE loss
#         # Video-to-Text: For each video, positive is its own text (diagonal), negatives are other texts
#         labels = torch.arange(batch_size, device=device)

#         # Text→Video loss
#         contrastive_loss = F.cross_entropy(similarity_matrix.T, labels, reduction='mean')

#         return contrastive_loss

#     def compute_diversity_loss(self, latent_tokens):
#         """
#         Compute diversity loss to encourage orthogonal latent token representations.

#         Forces different latent tokens to encode different aspects of the video
#         by penalizing high cosine similarity between tokens.

#         Args:
#             latent_tokens: [batch_size, num_latent_tokens, width] - Encoder output before quantization

#         Returns:
#             diversity_loss: Scalar tensor - Orthogonality penalty
#         """
#         B, K, D = latent_tokens.shape

#         # Normalize latent tokens for cosine similarity
#         latent_norm = F.normalize(latent_tokens, p=2, dim=-1)  # [B, K, D]

#         # Compute Gram matrix (cosine similarity between all token pairs)
#         # [B, K, D] × [B, D, K] → [B, K, K]
#         gram_matrix = torch.bmm(latent_norm, latent_norm.transpose(1, 2))

#         # Create identity matrix (target: orthogonal = diagonal only)
#         identity = torch.eye(K, device=gram_matrix.device, dtype=gram_matrix.dtype)
#         identity = identity.unsqueeze(0)  # [1, K, K]

#         # Penalize off-diagonal elements (want them to be 0)
#         # MSE loss: pushes Gram matrix towards identity
#         diversity_loss = F.mse_loss(gram_matrix, identity, reduction='mean')

#         return diversity_loss

#     def compute_loss(self, recon_video_features, quant_loss, video_features=None, encoder_out=None):
#         loss_recon = 0.0

#         # MSE Loss (weight index 0)
#         if self.vid_loss_weight[0] > 0:
#             loss_mse = F.mse_loss(recon_video_features, video_features, reduction='mean')
#             loss_recon += self.vid_loss_weight[0] * loss_mse

#         # Cosine Loss (weight index 2)
#         if self.vid_loss_weight[2] > 0:
#             loss_cosine = (1 - F.cosine_similarity(recon_video_features, video_features, dim=-1)).mean()
#             loss_recon += self.vid_loss_weight[2] * loss_cosine

#         # Diversity loss (if enabled and encoder_out provided)
#         loss_diversity = torch.tensor(0.0, device=recon_video_features.device)
#         if (encoder_out is not None and
#             hasattr(self, 'diversity_loss_weight') and
#             self.diversity_loss_weight > 0 and
#             self.num_latent_tokens > 1):
#             loss_diversity = self.compute_diversity_loss(encoder_out)
#             loss_diversity = self.diversity_loss_weight * loss_diversity

#         loss_total = loss_recon + self.quant_loss_weight * quant_loss + loss_diversity

#         return loss_total, loss_recon, loss_diversity