import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualAttentionBlock, ResidualCrossAttentionBlock, ResidualSelfCrossAttentionBlock
from .layers import MLPLayers

class VideoSlotMLPDecoder(nn.Module):
    """
    Slot-based MLP decoder for reconstructing video patch features from slot representations.

    Args:
        output_dim: Video patch feature dimension (e.g., 4096 for InternVL)
        num_patches: Number of video patches to reconstruct (e.g., 16, 128, 256)
        width: MLP decoder hidden dimension (default: 768)
        num_layers: Number of MLP hidden layers (default: 3)
        num_latent_tokens: Number of slots from encoder (default: 4)
        token_size: Input dimension of quantized tokens (default: None, uses width)
        decoder_input_dim: Optional intermediate projection dimension (default: None)

    Input:
        z_quantized: [batch_size, num_latent_tokens, token_size] - Quantized slot features

    Output:
        reconstructed_patches: [batch_size, num_patches, output_dim] - Reconstructed video features
    """

    def __init__(self,
                 output_dim=4096,           # Video patch feature dimension to reconstruct
                 num_patches=256,           # Number of video patches to reconstruct
                 width=768,                 # MLP decoder hidden dimension
                 num_layers=3,              # Number of MLP hidden layers
                 num_latent_tokens=4,       # Number of slots from encoder
                 token_size=None,           # Input dimension of quantized tokens
                 decoder_input_dim=None):   # Optional intermediate projection dimension
        super().__init__()

        self.output_dim = output_dim
        self.num_patches = num_patches
        self.width = width
        self.num_layers = num_layers
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size if token_size is not None else width
        self.decoder_input_dim = decoder_input_dim if decoder_input_dim is not None else width

        # 1. Input projection: token_size → decoder_input_dim (if dimensions differ)
        if self.token_size != self.decoder_input_dim:
            self.inp_transform = nn.Linear(self.token_size, self.decoder_input_dim, bias=True)
            nn.init.xavier_uniform_(self.inp_transform.weight)
            nn.init.zeros_(self.inp_transform.bias)
        else:
            self.inp_transform = None

        # 2. Dual positional embeddings
        scale = self.decoder_input_dim ** -0.5

        # Patch positional embeddings: temporal/spatial position information
        self.patch_pos_embed = nn.Parameter(
            torch.randn(1, num_patches, self.decoder_input_dim) * scale
        )

        # Slot positional embeddings: semantic role identifiers
        self.slot_pos_embed = nn.Parameter(
            torch.randn(1, num_latent_tokens, self.decoder_input_dim) * scale
        )

        # 3. MLP Decoder: (decoder_input_dim) → hidden layers → (output_dim + 1)
        # Build multi-layer MLP with ReLU activations and LayerNorm
        decoder_layers = []

        # Input layer with LayerNorm
        decoder_layers.append(nn.LayerNorm(self.decoder_input_dim))

        # Hidden layers
        current_dim = self.decoder_input_dim
        for i in range(num_layers):
            if i < num_layers - 1:
                # Hidden layer
                next_dim = self.decoder_input_dim * 2 if i == 0 else self.decoder_input_dim
                decoder_layers.append(nn.Linear(current_dim, next_dim))
                nn.init.xavier_uniform_(decoder_layers[-1].weight)
                nn.init.zeros_(decoder_layers[-1].bias)
                decoder_layers.append(nn.ReLU(inplace=True))
                decoder_layers.append(nn.LayerNorm(next_dim))
                current_dim = next_dim
            else:
                # Output layer: output_dim + 1 (features + alpha channel)
                decoder_layers.append(nn.Linear(current_dim, output_dim + 1))
                nn.init.xavier_uniform_(decoder_layers[-1].weight)
                nn.init.zeros_(decoder_layers[-1].bias)

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z_quantized):
        """
        Slot-based reconstruction of video patch features.

        Args:
            z_quantized: Quantized slot features from VideoSlotEncoder + RQ-VAE
                        Shape: [batch_size, num_latent_tokens, token_size]

        Returns:
            reconstructed_patches: Reconstructed video patch features
                                  Shape: [batch_size, num_patches, output_dim]

        Intermediate Shapes:
            - slots: [B, K, D]  (K = num_latent_tokens, D = decoder_input_dim)
            - slots_expanded: [B, K, P, D]  (P = num_patches)
            - slots_with_pos: [B, K, P, D]  (after adding positional embeddings)
            - outputs: [B, K, P, output_dim+1]  (decoded features + alpha)
            - features: [B, K, P, output_dim]
            - alpha: [B, K, P]  (normalized via softmax over K)
            - reconstructed: [B, P, output_dim]
        """
        # 1. Optional input projection: [B, K, token_size] → [B, K, decoder_input_dim]
        if self.inp_transform is not None:
            slots = self.inp_transform(z_quantized)  # [B, K, D]
        else:
            slots = z_quantized  # [B, K, D]

        # 2. Add slot positional embeddings (semantic roles)
        # [B, K, D] + [1, K, D] → [B, K, D]
        slots = slots + self.slot_pos_embed.to(slots.dtype)

        # 3. Expand slots to patch dimension: [B, K, D] → [B, K, P, D]
        # Each slot will generate features for all patches
        slots_expanded = slots.unsqueeze(2).expand(-1, -1, self.num_patches, -1)

        # 4. Add patch positional embeddings (temporal/spatial positions)
        # [B, K, P, D] + [1, 1, P, D] → [B, K, P, D]
        slots_with_pos = slots_expanded + self.patch_pos_embed.unsqueeze(1).to(slots_expanded.dtype)

        # 5. Flatten for MLP processing: [B, K, P, D] → [B*K*P, D]
        initial_shape = slots_with_pos.shape[:3]  # [B, K, P]
        slots_flat = slots_with_pos.reshape(-1, self.decoder_input_dim)

        # 6. Apply MLP decoder: [B*K*P, D] → [B*K*P, output_dim+1]
        output_flat = self.decoder(slots_flat)

        # 7. Reshape back: [B*K*P, output_dim+1] → [B, K, P, output_dim+1]
        output = output_flat.reshape(*initial_shape, -1)

        # 8. Split into patch features and alpha channel
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)
        # decoded_patches: [B, K, P, output_dim]
        # alpha: [B, K, P, 1]

        # 9. Normalize alpha masks with softmax over slots (competitive assignment)
        alpha = alpha.squeeze(-1)  # [B, K, P]
        alpha = F.softmax(alpha, dim=1)  # Softmax over slot dimension

        # 10. Compositional reconstruction: weighted sum over slots
        # decoded_patches: [B, K, P, output_dim]
        # alpha: [B, K, P] → [B, K, P, 1]
        alpha = alpha.unsqueeze(-1)
        reconstruction = torch.sum(decoded_patches * alpha, dim=1)  # [B, P, output_dim]

        return reconstruction


class VideoDecoder(nn.Module):
    def __init__(self,
                 output_dim=4096,           # Video patch feature dimension to reconstruct
                 num_patches=256,           # Number of video patches to reconstruct
                 width=768,                 # Decoder transformer width
                 num_layers=6,              # Number of transformer layers
                 num_heads=12,              # Number of attention heads
                 num_latent_tokens=1,       # Number of latent tokens from encoder
                 token_size=None):          # Input dimension of quantized tokens (defaults to width)
        super().__init__()

        self.output_dim = output_dim
        self.num_patches = num_patches
        self.width = width
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size if token_size is not None else width

        # Project quantized latent tokens to decoder width
        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        scale = self.width ** -0.5

        # Mask tokens for video patch reconstruction (256 learnable tokens)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))

        # Dual positional embedding strategy:
        # 1. Mask tokens: Temporal position embeddings for video patches (0-255)
        self.mask_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_patches, self.width))

        # 2. Latent tokens: Semantic role embeddings separate from temporal positions
        self.latent_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        # For video output: linear layer to reconstruct video features
        self.video_output_proj = nn.Linear(self.width, self.output_dim)
    
    def forward(self, z_quantized):
        """
        MAE-style video reconstruction decoder.
        Args:
            z_quantized: Quantized latent tokens from ResidualVectorQuantizer
                        Shape: [batch_size, num_latent_tokens, width]
        Returns:
            reconstructed_patches: [batch_size, num_patches, output_dim]
        """
        batch_size = z_quantized.shape[0]

        # 1. Project quantized latent tokens to decoder width
        # z_quantized: [batch_size, num_latent_tokens, token_size] -> [batch_size, num_latent_tokens, width]
        latent_tokens = self.decoder_embed(z_quantized)

        # 2. Initialize mask tokens for video patch reconstruction
        # mask_token: [1, 1, width] -> [batch_size, num_patches, width]
        mask_tokens = self.mask_token.repeat(batch_size, self.num_patches, 1).to(latent_tokens.dtype)

        # 3. Add positional embeddings separately
        # Mask tokens: temporal position embeddings (which video patch position)
        mask_tokens = mask_tokens + self.mask_positional_embedding.to(mask_tokens.dtype)

        # Latent tokens: semantic role embeddings (their function in summarization)
        latent_tokens = latent_tokens + self.latent_positional_embedding.to(latent_tokens.dtype)

        # 4. Concatenate: [mask_tokens, latent_tokens]
        # Shape: [batch_size, num_patches + num_latent_tokens, width]
        x = torch.cat([mask_tokens, latent_tokens], dim=1)

        # 5. Apply transformer self-attention layers
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND for transformer
        for layer in self.transformer:
            x = layer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 6. Extract only mask token outputs (first num_patches tokens)
        # These contain the reconstructed video patch features
        mask_outputs = x[:, :self.num_patches]  # [batch_size, num_patches, width]
        mask_outputs = self.ln_post(mask_outputs)

        # 7. Project to original video feature dimension
        # [batch_size, num_patches, width] -> [batch_size, num_patches, output_dim]
        reconstructed_patches = self.video_output_proj(mask_outputs)

        return reconstructed_patches


class VideoLatentDecoder(nn.Module):
    """
    Unlike VideoDecoder which concatenates mask_tokens and latent_tokens for joint self-attention,
    this decoder uses separate self-attention and cross-attention in each transformer block.

    Architecture (per layer):
    1. Self-attention: mask_tokens attend to themselves (spatial/temporal modeling)
    2. Cross-attention: mask_tokens attend to latent_tokens (semantic conditioning)
    3. Feedforward MLP: non-linear transformation

    Args:
        output_dim: Video patch feature dimension to reconstruct (e.g., 4096 for InternVL)
        num_patches: Number of video patches to reconstruct (e.g., 16, 128, 256)
        width: Decoder transformer width
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_latent_tokens: Number of latent tokens from encoder
        token_size: Input dimension of quantized tokens (defaults to width)
    """
    def __init__(self,
                 output_dim=4096,           # Video patch feature dimension to reconstruct
                 num_patches=256,           # Number of video patches to reconstruct
                 width=768,                 # Decoder transformer width
                 num_layers=6,              # Number of transformer layers
                 num_heads=12,              # Number of attention heads
                 num_latent_tokens=1,       # Number of latent tokens from encoder
                 token_size=None):          # Input dimension of quantized tokens (defaults to width)
        super().__init__()

        self.output_dim = output_dim
        self.num_patches = num_patches
        self.width = width
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size if token_size is not None else width

        # Project quantized latent tokens to decoder width
        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        scale = self.width ** -0.5

        # Mask tokens for video patch reconstruction (256 learnable tokens)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))

        # Dual positional embedding strategy:
        # 1. Mask tokens: Temporal position embeddings for video patches (0-255)
        self.mask_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_patches, self.width))

        # 2. Latent tokens: Semantic role embeddings separate from temporal positions
        self.latent_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))

        self.ln_pre = nn.LayerNorm(self.width)

        # Self + Cross-attention transformer layers (Stable Diffusion 1.4 style)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualSelfCrossAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))

        self.ln_post = nn.LayerNorm(self.width)

        # For video output: linear layer to reconstruct video features
        self.video_output_proj = nn.Linear(self.width, self.output_dim)

    def forward(self, z_quantized):
        """
        Self + Cross-attention based video reconstruction decoder (Stable Diffusion style).

        Args:
            z_quantized: Quantized latent tokens from ResidualVectorQuantizer
                        Shape: [batch_size, num_latent_tokens, token_size]

        Returns:
            reconstructed_patches: [batch_size, num_patches, output_dim]
        """
        batch_size = z_quantized.shape[0]

        # 1. Project quantized latent tokens to decoder width
        # z_quantized: [batch_size, num_latent_tokens, token_size] -> [batch_size, num_latent_tokens, width]
        latent_tokens = self.decoder_embed(z_quantized)

        # 2. Initialize mask tokens for video patch reconstruction
        # mask_token: [1, 1, width] -> [batch_size, num_patches, width]
        mask_tokens = self.mask_token.repeat(batch_size, self.num_patches, 1).to(latent_tokens.dtype)

        # 3. Add positional embeddings separately
        # Mask tokens: temporal position embeddings (which video patch position)
        mask_tokens = mask_tokens + self.mask_positional_embedding.to(mask_tokens.dtype)

        # # Latent tokens: semantic role embeddings (their function in summarization)
        # latent_tokens = latent_tokens + self.latent_positional_embedding.to(latent_tokens.dtype)

        # 5. Apply self + cross-attention transformer layers 
        # Note: ResidualSelfCrossAttentionBlock expects [seq_len, batch_size, width] format
        query = self.ln_pre(mask_tokens)
        query = query.permute(1, 0, 2)      # NLD -> LND (num_patches, batch_size, width)
        latent_tokens = latent_tokens.permute(1, 0, 2)  # NLD -> LND (num_latent_tokens, batch_size, width)

        for layer in self.transformer:
            # Each layer: self-attention -> cross-attention -> MLP
            query = layer(query, latent_tokens)

        query = query.permute(1, 0, 2)  # LND -> NLD (batch_size, num_patches, width)

        # 6. Output is directly the query (no slicing needed)
        query = self.ln_post(query)  # [batch_size, num_patches, width]

        # 7. Project to original video feature dimension
        # [batch_size, num_patches, width] -> [batch_size, num_patches, output_dim]
        reconstructed_patches = self.video_output_proj(query)

        return reconstructed_patches


class VideoLatentDecoder_V2(nn.Module):
    """
    MLP-based decoder that reconstructs pooled video features from disentangled slot representations.
    Mirrors VideoLatentEncoder_V2 architecture with reversed MLPs.

    This decoder:
    1. Takes quantized slot tokens [B, num_latent_tokens, token_size]
    2. Uses parallel MLPs (one per latent token) to decode each slot independently
    3. Averages decoded features to produce final reconstruction [B, output_dim]

    Architecture:
    - Input: Quantized slots from encoder [B, num_latent_tokens, token_size]
    - Processing: Each slot decoded by its own MLP with reversed layer dimensions
    - Outputs: Individual decoded features AND averaged reconstruction
    """
    def __init__(
        self,
        output_dim,
        num_latent_tokens,
        mlp_layers,
        mlp_dropout=0.0,
        mlp_activation="relu",
        mlp_bn=False
    ):
        super().__init__()

        self.output_dim = output_dim
        self.num_latent_tokens = num_latent_tokens
        self.mlp_layers = mlp_layers
        
        # Create one MLP per latent token 
        self.mlps = nn.ModuleList([
            MLPLayers(
                layers=self.mlp_layers,
                dropout=mlp_dropout,
                activation=mlp_activation,
                bn=mlp_bn
            )
            for _ in range(num_latent_tokens)
        ])

    def forward(self, z_quantized):
        """
        Reconstruct pooled video features from quantized slots via parallel MLPs.

        Args:
            z_quantized: [B, num_latent_tokens, token_size] - Quantized slot tokens from RQ-VAE

        Returns:
            decoded_features: [B, num_latent_tokens, output_dim] - Individual decoded partial features
            reconstructed_video: [B, output_dim] - Averaged reconstruction from all slots
        """
        # Each MLP decodes its corresponding latent token independently
        # z_quantized[:, i, :] -> mlps[i] -> [B, output_dim]
        decoded_list = []
        for i in range(self.num_latent_tokens):
            latent_token_i = z_quantized[:, i, :]  # [B, token_size]
            decoded_i = self.mlps[i](latent_token_i)  # [B, output_dim]
            decoded_list.append(decoded_i)

        # Stack individual decoded features: [B, num_latent_tokens, output_dim]
        decoded_features = torch.stack(decoded_list, dim=1)

        # Average across latent tokens for final reconstruction: [B, output_dim]
        reconstructed_video = decoded_features.mean(dim=1)

        return decoded_features, reconstructed_video


# class VideoLatentDecoder_V2(nn.Module):
#     """
#     Cross-attention based decoder that reconstructs pooled video features from slot representations.

#     This decoder mirrors VideoLatentEncoder_V2 architecture with reversed attention:
#     - Encoder: Slots (query) attend to video feature (key/value) to extract aspects
#     - Decoder: Video query (query) attends to slots (key/value) to reconstruct feature

#     Architecture:
#     1. Takes quantized slot tokens [B, num_latent_tokens, token_size]
#     2. Uses learnable video query to reconstruct via cross-attention
#     3. Produces reconstructed pooled video feature [B, output_dim]

#     Key Design:
#     - Query: Learnable video query (initialized with larger scale)
#     - Key/Value: Projected quantized slot tokens
#     - Attention: Cross-attention (video query attends to slots)
#     """
#     def __init__(self, output_dim=512, width=512, num_layers=1, num_heads=8, num_latent_tokens=4, token_size=64):
#         super().__init__()

#         self.output_dim = output_dim
#         self.width = width
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.num_latent_tokens = num_latent_tokens
#         self.token_size = token_size

#         # Quantized token projection
#         if self.token_size != self.width:
#             self.decoder_embed = nn.Linear(self.token_size, self.width)
#         else:
#             self.decoder_embed = nn.Identity()

#         # Initialize learnable video query for reconstruction
#         self.learnable_video_query = nn.Parameter(torch.empty(1, self.width))
#         self._init_video_query()

#         self.ln_pre = nn.LayerNorm(self.width)

#         # Cross-attention transformer layers (video query attends to slots)
#         self.transformer = nn.ModuleList()
#         for i in range(self.num_layers):
#             self.transformer.append(ResidualCrossAttentionBlock(
#                 d_model=self.width,
#                 n_head=self.num_heads,
#                 mlp_ratio=4.0,
#                 act_layer=nn.GELU,
#                 norm_layer=nn.LayerNorm
#             ))

#         self.ln_post = nn.LayerNorm(self.width)

#         # Project to original video feature dimension
#         if self.width != self.output_dim:
#             self.video_output_proj = nn.Linear(self.width, self.output_dim)
#         else:
#             self.video_output_proj = nn.Identity()

#     def _init_video_query(self):
#         """
#         Initialize video query with larger scale to promote effective reconstruction.
#         Uses same scale as encoder's latent token initialization (2.0 * width ** -0.5).
#         """
#         scale = 2.0 * self.width ** -0.5
#         nn.init.normal_(self.learnable_video_query, std=scale)

#     def forward(self, z_quantized):
#         """
#         Reconstruct pooled video features from quantized slots via cross-attention.

#         Args:
#             z_quantized: [batch_size, num_latent_tokens, token_size] - Quantized slot tokens from RQ-VAE

#         Returns:
#             reconstructed_video: [batch_size, output_dim] - Reconstructed pooled video feature
#         """
#         batch_size = z_quantized.shape[0]

#         # 1. Project quantized slots to decoder width
#         latent_tokens = self.decoder_embed(z_quantized)  # [B, num_latent_tokens, width]

#         # 2. Normalize slot context
#         latent_tokens = self.ln_pre(latent_tokens)  # [B, num_latent_tokens, width]

#         # 3. Initialize video query from learnable parameter
#         video_query = self.learnable_video_query.expand(batch_size, -1)  # [B, width]
#         video_query = video_query.unsqueeze(1)  # [B, 1, width]

#         # 4. Apply cross-attention layers
#         # ResidualCrossAttentionBlock expects [seq_len, batch, dim] format
#         video_query = video_query.permute(1, 0, 2)  # [1, B, width]
#         latent_tokens = latent_tokens.permute(1, 0, 2)  # [num_latent_tokens, B, width]

#         for layer in self.transformer:
#             video_query = layer(
#                 query=video_query,        # [1, B, width]
#                 context=latent_tokens     # [num_latent_tokens, B, width]
#             )

#         # 5. Permute back to batch-first format
#         video_query = video_query.permute(1, 0, 2)  # [B, 1, width]
#         video_query = video_query.squeeze(1)  # [B, width]

#         # 6. Final layer norm
#         video_query = self.ln_post(video_query)

#         # 7. Project to original video feature dimension
#         reconstructed_video = self.video_output_proj(video_query)  # [B, output_dim]

#         return reconstructed_video