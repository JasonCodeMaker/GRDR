import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualAttentionBlock, LatentTokenAttentionBlock, ResidualCrossAttentionBlock
from .layers import MLPLayers


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class VideoSlotEncoder(nn.Module):
    """
    Video encoder using full slot attention mechanism with iterative refinement.

    Based on "Object-Centric Learning with Slot Attention" (Locatello et al., 2020).
    Key features:
    - Gaussian slot initialization (learnable mu/sigma)
    - Iterative refinement with attention
    - GRU-based slot updates
    - MLP refinement per iteration

    This encoder replaces standard transformer encoding with slot attention,
    allowing slots to compete for explaining different aspects of the video.
    """

    def __init__(
        self,
        input_dim=4096,
        num_patches=128,
        width=768,
        num_iterations=3,
        num_latent_tokens=4,
        hidden_dim=128,
        eps=1e-8
    ):
        """
        Args:
            input_dim: Input video patch feature dimension
            num_patches: Number of video patches
            width: Slot dimension (encoder width)
            num_iterations: Number of slot attention iterations
            num_latent_tokens: Number of slots
            hidden_dim: Hidden dimension for slot MLP
            eps: Small constant for numerical stability
        """
        super().__init__()

        # Video feature input dimensions
        self.input_dim = input_dim
        self.num_patches = num_patches
        self.width = width
        self.hidden_dim = max(width, hidden_dim)

        # Video feature projection
        if self.input_dim != self.width:
            self.video_embed = nn.Linear(self.input_dim, self.width)
        else:
            self.video_embed = nn.Identity()

        # Positional embeddings for video patches only
        self.scale = self.width ** -0.5
        self.positional_embedding = nn.Parameter(
            self.scale * torch.randn(self.num_patches, self.width)
        )

        # Slot attention parameters
        self.num_slots = num_latent_tokens
        self.iters = num_iterations
        self.eps = eps

        # Gaussian slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.width))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, self.width)))

        # Attention Q/K/V projections
        self.to_q = nn.Linear(self.width, self.width)
        self.to_k = nn.Linear(self.width, self.width)
        self.to_v = nn.Linear(self.width, self.width)

        # GRU for slot updates
        self.gru = nn.GRUCell(self.width, self.width)

        # MLP for slot refinement
        self.fc1 = nn.Linear(self.width, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.width)

        # Layer normalizations
        self.norm_input = nn.LayerNorm(self.width)
        self.norm_slots = nn.LayerNorm(self.width)
        self.norm_pre_ff = nn.LayerNorm(self.width)

    def forward(self, pixel_values):
        """
        Forward pass with iterative slot attention.

        Args:
            pixel_values: [batch_size, num_patches, input_dim] - Video patch features

        Returns:
            slots: [batch_size, num_latent_tokens, width] - Refined slot features
        """
        batch_size, num_patches, _ = pixel_values.shape

        # 1. Project video features and add positional embeddings
        inputs = self.video_embed(pixel_values)  # [B, N, width]
        inputs = inputs + self.positional_embedding.to(inputs.dtype)

        # 2. Normalize inputs
        inputs = self.norm_input(inputs)  # [B, N, width]

        # 3. Compute keys and values from inputs (done once)
        k = self.to_k(inputs)  # [B, N, width]
        v = self.to_v(inputs)  # [B, N, width]

        # 4. Initialize slots from Gaussian distribution
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_sigma.expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)  # [B, num_slots, width]

        # 5. Iterative slot attention refinement
        for _ in range(self.iters):
            slots_prev = slots

            # Normalize slots
            slots_norm = self.norm_slots(slots)  # [B, num_slots, width]

            # Compute queries from slots
            q = self.to_q(slots_norm)  # [B, num_slots, width]

            # Attention: slots attend to input patches
            # dots: [B, num_slots, N]
            dots = torch.einsum('bsd,bnd->bsn', q, k) * self.scale

            # Normalize over slots (competition among slots for inputs)
            attn = F.softmax(dots, dim=1) + self.eps  # [B, num_slots, N]

            # Normalize over inputs (weighted mean)
            attn = attn / attn.sum(dim=-1, keepdim=True)  # [B, num_slots, N]

            # Weighted sum of values
            updates = torch.einsum('bnd,bsn->bsd', v, attn)  # [B, num_slots, width]

            # GRU update
            # Reshape for GRU: [B*num_slots, width]
            slots = self.gru(
                updates.reshape(-1, self.width),
                slots_prev.reshape(-1, self.width)
            )
            slots = slots.reshape(batch_size, self.num_slots, self.width)

            # MLP refinement
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots


class VideoEncoder(nn.Module):
    def __init__(self, input_dim=4096, num_patches=256, width=768, num_layers=6, num_heads=12, num_latent_tokens=1, token_size=64):
        super().__init__()

        # Video feature input dimensions
        self.input_dim = input_dim
        self.num_patches = num_patches
        self.width = width
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        
        # For video features: linear projection if dimensions don't match
        if self.input_dim != self.width:
            self.video_embed = nn.Linear(self.input_dim, self.width)
        else:
            self.video_embed = nn.Identity()

        scale = self.width ** -0.5

        # Positional embeddings for video features and latent tokens
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.num_patches, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))

        # Initialize learnable latent tokens
        self.learnable_latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.width))
        
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

    def forward(self, pixel_values, latent_tokens):
        # Handle video features: (batch_size, num_patches, input_dim) -> (batch_size, num_patches, width)
        x = self.video_embed(pixel_values)  # Shape: (batch_size, num_patches, width)

        # Add positional embeddings to video patches (no class token)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, num_patches, width]

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)

        # Concatenate video patches with latent tokens
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Extract latent tokens (last num_latent_tokens positions)
        latent_tokens = x[:, self.num_patches:]
        latent_tokens = self.ln_post(latent_tokens)

        # Return latent tokens: (batch_size, num_latent_tokens, width)
        return latent_tokens


class VideoLatentEncoder(nn.Module):
    """
    Slot-based video encoder where latent tokens operate independently without mutual attention.

    Key differences from VideoEncoder:
    1. No positional encoding for latent tokens
    2. Latent tokens cannot attend to each other (slot attention)
    3. Orthogonal initialization for latent tokens
    """
    def __init__(self, input_dim=4096, num_patches=256, width=768, num_layers=6, num_heads=12, num_latent_tokens=1, token_size=64, version="1.1"):
        super().__init__()

        self.input_dim = input_dim
        self.num_patches = num_patches
        self.width = width
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size

        # Video feature projection
        if self.input_dim != self.width:
            self.video_embed = nn.Linear(self.input_dim, self.width)
        else:
            self.video_embed = nn.Identity()

        scale = self.width ** -0.5

        # Positional embeddings only for video patches (not for latent tokens)
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.num_patches, self.width))

        if version == "1.1":
            # Initialize learnable latent tokens with orthogonal initialization
            self.learnable_latent_tokens = nn.Parameter(torch.empty(self.num_latent_tokens, self.width))
            self._init_latent_tokens_orthogonal()
        else:
            self.learnable_latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.width))

        self.ln_pre = nn.LayerNorm(self.width)

        # Use LatentTokenAttentionBlock instead of ResidualAttentionBlock
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(LatentTokenAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0,
                num_patches=self.num_patches,
                num_latent_tokens=self.num_latent_tokens
            ))

        self.ln_post = nn.LayerNorm(self.width)

    def _init_latent_tokens_orthogonal(self):
        """
        Initialize latent tokens with orthogonal initialization for maximum diversity.
        Uses larger scale (2.0 * width ** -0.5) to promote initial separation.
        """
        scale = 2.0 * self.width ** -0.5

        if self.num_latent_tokens == 1:
            # Single token: use standard normal initialization
            nn.init.normal_(self.learnable_latent_tokens, std=scale)
        else:
            # Multiple tokens: orthogonal initialization
            # Generate orthogonal matrix of shape [num_latent_tokens, width]
            orthogonal_matrix = torch.nn.init.orthogonal_(
                torch.empty(self.num_latent_tokens, self.width)
            )
            # Scale to desired magnitude
            self.learnable_latent_tokens.data = orthogonal_matrix * scale

    def forward(self, pixel_values, latent_tokens):
        """
        Forward pass with slot attention mechanism.

        Args:
            pixel_values: [batch_size, num_patches, input_dim]
            latent_tokens: [num_latent_tokens, width] - learnable parameters

        Returns:
            latent_tokens: [batch_size, num_latent_tokens, width]
        """
        # Project video features
        x = self.video_embed(pixel_values)

        # Add positional embeddings only to video patches
        x = x + self.positional_embedding.to(x.dtype)

        # Expand latent tokens for batch (NO positional encoding added)
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)

        # Concatenate video patches with latent tokens
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Apply slot attention blocks
        for i in range(self.num_layers):
            x = self.transformer[i](x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        # Extract latent tokens (last num_latent_tokens positions)
        latent_tokens = x[:, self.num_patches:]
        latent_tokens = self.ln_post(latent_tokens)

        return latent_tokens

class VideoLatentEncoder_V2(nn.Module):
    """
    MLP-based encoder that disentangles pooled video features into multiple partial features.
    Uses parallel MLPs (one per latent token) to process the same input independently.
    """
    def __init__(
        self,
        input_dim,
        num_latent_tokens,
        mlp_layers,
        mlp_dropout=0.0,
        mlp_activation="relu",
        mlp_bn=False
    ):
        super().__init__()

        self.input_dim = input_dim
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

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [B, input_dim] - Pooled video feature

        Returns:
            outputs: [B, num_latent_tokens, output_dim] - Disentangled partial features
        """
        # Each MLP processes the same input independently
        outputs = [mlp(pixel_values) for mlp in self.mlps]  # num_latent_tokens × [B, output_dim]

        # Stack along latent token dimension
        return torch.stack(outputs, dim=1)  # [B, num_latent_tokens, output_dim]



# class VideoLatentEncoder_V2(nn.Module):
#     """
#     Self-attention based encoder that disentangles pooled video features into slot representations.

#     This encoder:
#     1. Takes a single pooled video feature [B, input_dim]
#     2. Uses learnable slot queries to extract different aspects via self-attention
#     3. Produces disentangled slot representations [B, num_latent_tokens, width]

#     Architecture:
#     - Input: Pooled video feature [B, input_dim]
#     - Query: Learnable slot tokens (initialized orthogonally) [num_latent_tokens, B, width]
#     - Attention: Self-attention (slots attend to each other)
#     - Output: Disentangled slots [B, num_latent_tokens, width]

#     Args:
#         input_dim: Input feature dimension
#         width: Model width (feature dimension)
#         num_layers: Number of self-attention layers
#         num_heads: Number of attention heads
#         num_latent_tokens: Number of slot tokens (output slots)
#         token_size: For API compatibility (unused)
#     """
#     def __init__(
#         self,
#         input_dim=512,
#         width=512,
#         num_layers=1,
#         num_heads=8,
#         num_latent_tokens=4,
#         token_size=64
#     ):
#         super().__init__()

#         self.input_dim = input_dim
#         self.width = width
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.num_latent_tokens = num_latent_tokens
#         self.token_size = token_size  # For API compatibility (unused)

#         # Video feature projection
#         if self.input_dim != self.width:
#             self.video_embed = nn.Linear(self.input_dim, self.width)
#         else:
#             self.video_embed = nn.Identity()

#         # Initialize learnable latent tokens with orthogonal initialization
#         self.learnable_latent_tokens = nn.Parameter(torch.empty(self.num_latent_tokens, self.width))
#         self._init_latent_tokens_orthogonal()

#         self.ln_pre = nn.LayerNorm(self.width)

#         self.transformer = nn.ModuleList()
#         for i in range(self.num_layers):
#             self.transformer.append(ResidualAttentionBlock(
#                 self.width, self.num_heads, mlp_ratio=4.0
#             ))

#         self.ln_post = nn.LayerNorm(self.width)

#     def _init_latent_tokens_orthogonal(self):
#         """
#         Initialize latent tokens with orthogonal initialization for maximum diversity.
#         Uses larger scale (2.0 * width ** -0.5) to promote initial separation.
#         """
#         scale = 2.0 * self.width ** -0.5

#         if self.num_latent_tokens == 1:
#             # Single token: use standard normal initialization
#             nn.init.normal_(self.learnable_latent_tokens, std=scale)
#         else:
#             # Multiple tokens: orthogonal initialization
#             # Generate orthogonal matrix of shape [num_latent_tokens, width]
#             orthogonal_matrix = torch.nn.init.orthogonal_(
#                 torch.empty(self.num_latent_tokens, self.width)
#             )
#             # Scale to desired magnitude
#             self.learnable_latent_tokens.data = orthogonal_matrix * scale

#     def forward(self, pixel_values, latent_tokens):
#         """
#         Args:
#             pixel_values: [B, input_dim] - Pooled video feature (NOT patches!)
#             latent_tokens: [num_latent_tokens, width] - Learnable slot queries

#         Returns:
#             latent_tokens: [B, num_latent_tokens, width] - Disentangled slots
#         """

#         # 1. Project video features to encoder width
#         x = self.video_embed(pixel_values)  # [B, input_dim] -> [B, width]
#         x = x.unsqueeze(1)  # [B, width] -> [B, 1, width]

#         # 2. Expand latent tokens for batch (no permutation needed yet)
#         latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)  # [B, num_latent_tokens, width]

#         # 3. Concatenate single context token with latent tokens
#         x = torch.cat([x, latent_tokens], dim=1)  # [B, 1+num_latent_tokens, width]

#         # 4. Apply layer norm ONCE before transformer
#         x = self.ln_pre(x)  # [B, 1+num_latent_tokens, width]

#         # 5. Permute for transformer (expects LND format)
#         x = x.permute(1, 0, 2)  # [B, 1+num_latent_tokens, width] -> [1+num_latent_tokens, B, width]

#         # 6. Apply transformer layers
#         for i in range(self.num_layers):
#             x = self.transformer[i](x)

#         # 7. Permute back to batch-first
#         x = x.permute(1, 0, 2)  # [1+num_latent_tokens, B, width] -> [B, 1+num_latent_tokens, width]

#         # 8. Extract latent tokens (skip the first context token)
#         latent_tokens = x[:, 1:]  # [B, num_latent_tokens, width]
#         latent_tokens = self.ln_post(latent_tokens)

#         # Return latent tokens: (batch_size, num_latent_tokens, width)
#         return latent_tokens
