import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class LatentTokenAttentionBlock(nn.Module):
    """
    Custom attention block where latent tokens (slots) can only attend to input patches,
    not to each other. Input patches can attend to all tokens freely.
    """
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio=4.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            num_patches=256,
            num_latent_tokens=1
        ):
        super().__init__()
        self.num_patches = num_patches
        self.num_latent_tokens = num_latent_tokens

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio

        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

        self._register_attention_mask()

    def _register_attention_mask(self):
        """
        Create attention mask to prevent latent tokens from attending to each other.
        Mask shape: [num_patches + num_latent_tokens, num_patches + num_latent_tokens]

        Attention pattern (False = allow, True = block):
        - Patches → Patches: False (allowed)
        - Patches → Latents: False (allowed)
        - Latents → Patches: False (allowed)
        - Latents → Latents: True (blocked)
        """
        total_len = self.num_patches + self.num_latent_tokens
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        # Block latent-to-latent attention
        # Latent tokens are at positions [num_patches : num_patches + num_latent_tokens]
        latent_start = self.num_patches
        latent_end = self.num_patches + self.num_latent_tokens
        mask[latent_start:latent_end, latent_start:latent_end] = True

        # Allow self-attention for each individual latent token
        for i in range(self.num_latent_tokens):
            mask[latent_start + i, latent_start + i] = False

        self.register_buffer('attn_mask', mask)

    def attention(self, x: torch.Tensor):
        """Apply masked self-attention."""
        return self.attn(x, x, x, attn_mask=self.attn_mask, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio

        # MLP block (optional, controlled by mlp_ratio)
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            query: torch.Tensor,
            context: torch.Tensor
    ):
        return self.attn(query, context, context, need_weights=False)[0]

    def forward(
            self,
            query: torch.Tensor,
            context: torch.Tensor
    ):
        # Cross-attention with residual connection
        attn_output = self.attention(query=self.ln_1(query), context=context)
        query = query + attn_output

        # MLP with residual connection (if enabled)
        if self.mlp_ratio > 0:
            query = query + self.mlp(self.ln_2(query))

        return query


class ResidualSelfCrossAttentionBlock(nn.Module):
    """
    Architecture:
    1. Self-attention: queries attend to themselves (intra-query communication)
    2. Cross-attention: queries attend to context/latent tokens
    3. Feedforward MLP: non-linear transformation

    Each sub-layer uses pre-normalization (LayerNorm before attention/MLP)
    with residual connections, following the standard transformer design.

    Args:
        d_model: Model dimension (hidden size)
        n_head: Number of attention heads
        mlp_ratio: MLP hidden dimension multiplier (hidden_dim = d_model * mlp_ratio)
        act_layer: Activation function for MLP (default: GELU)
        norm_layer: Normalization layer type (default: LayerNorm)
    """
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        # Layer normalization for each sub-layer (pre-norm architecture)
        self.ln_1 = norm_layer(d_model)  # for self-attention
        self.ln_2 = norm_layer(d_model)  # for cross-attention
        self.ln_3 = norm_layer(d_model)  # for MLP

        # Self-attention: queries attend to themselves
        self.self_attn = nn.MultiheadAttention(d_model, n_head)

        # Cross-attention: queries attend to context
        self.cross_attn = nn.MultiheadAttention(d_model, n_head)

        self.mlp_ratio = mlp_ratio

        # MLP block (optional, controlled by mlp_ratio)
        if mlp_ratio > 0:
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def forward(
            self,
            query: torch.Tensor,
            context: torch.Tensor
    ):
        """
        Forward pass following Stable Diffusion's transformer block pattern.

        Args:
            query: Query tensor [seq_len, batch_size, d_model]
            context: Context tensor for cross-attention [context_len, batch_size, d_model]

        Returns:
            query: Transformed query tensor [seq_len, batch_size, d_model]
        """
        # 1. Self-attention: queries attend to themselves
        # Allows mask tokens to exchange spatial/temporal information
        self_attn_output = self.self_attn(
            query=self.ln_1(query),
            key=self.ln_1(query),
            value=self.ln_1(query),
            need_weights=False
        )[0]
        query = query + self_attn_output

        # 2. Cross-attention: queries attend to context (latent tokens)
        # Pulls semantic information from quantized video embeddings
        cross_attn_output = self.cross_attn(
            query=self.ln_2(query),
            key=context,
            value=context,
            need_weights=False
        )[0]
        query = query + cross_attn_output

        # 3. MLP: non-linear transformation with residual connection
        if self.mlp_ratio > 0:
            query = query + self.mlp(self.ln_3(query))

        return query


if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TextDecoder(nn.Module):
    """
    TextDecoder: Reconstructs text embeddings from selected quantized video embeddings.

    Follows the RQVAE decoder pattern using MLPLayers to transform from quantized space
    back to text embedding space.
    """

    def __init__(self,
                 input_dim=768,          # e_dim from selected quantized token
                 text_dim=4096,          # target text embedding dimension
                 hidden_layers=None,    # intermediate layers
                 dropout_prob=0.0,      # dropout probability
                 bn=False):             # batch normalization
        """
        Args:
            input_dim: Dimension of input quantized embedding (e_dim)
            text_dim: Target text embedding dimension
            hidden_layers: List of hidden layer sizes, default [256, 512]
            dropout_prob: Dropout probability for MLPLayers
            bn: Whether to use batch normalization
        """
        super(TextDecoder, self).__init__()

        self.input_dim = input_dim
        self.text_dim = text_dim
        self.dropout_prob = dropout_prob
        self.bn = bn

        # Default hidden layers if not specified
        if hidden_layers is None:
            hidden_layers = [1536, 3072]

        # Import MLPLayers from layers module
        from .layers import MLPLayers

        # Build decoder layers: [input_dim] + hidden_layers + [text_dim]
        self.decode_layer_dims = [input_dim] + hidden_layers + [text_dim]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                               dropout=dropout_prob,
                               bn=bn)

    def forward(self, selected_q_emb):
        """
        Forward pass to reconstruct text embedding from selected quantized video embedding.

        Args:
            selected_q_emb: Selected quantized embedding [batch_size, input_dim]

        Returns:
            reconstructed_text: Reconstructed text embedding [batch_size, text_dim]
        """
        return self.decoder(selected_q_emb)

class LatentTokenRouter(nn.Module):
    """
    Enhanced context-aware router with soft mixture of experts.

    Architecture:
    - Projects text embeddings to quantized token space (e_dim)
    - Computes cosine similarity between text and all quantized tokens
    - Forms soft weighted mixture of tokens using gate probabilities
    - Provides backward compatibility with discrete token indices

    Key Features:
    - Actual routed embedding is soft mixture via torch.einsum('be,bed->bd')
    - Gate probabilities represent contribution weights, not selection probabilities
    - Argmax indices available for compatibility with semantic ID tooling
    - Temperature parameters control similarity computation and mixture characteristics
    - Optional noisy gating for training-time exploration
    """

    def __init__(self,
                 text_dim=4096,         # input text embedding dim
                 e_dim=768,              # quantized token dim (matches q_video_emb's last dim)
                 num_latent_tokens=1,   # number of experts/latent tokens
                 temperature=1.0,       # softmax temperature for similarity computation
                 soft_temperature=1.0,  # additional temperature for mixture sharpening/flattening
                 noisy_gating=True,    # add input-dependent noise during training
                 noise_epsilon=1e-2):   # small constant for noise numerical stability
        super().__init__()
        self.text_dim = text_dim
        self.e_dim = e_dim
        self.num_latent_tokens = num_latent_tokens
        self.temperature = temperature
        self.soft_temperature = soft_temperature
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon

        # Normalize text first; we keep text tower frozen (detach in forward)
        self.text_norm = nn.LayerNorm(text_dim)

        # Single-side projection: text -> e_dim (no q_proj)
        self.t_proj = nn.Linear(text_dim, e_dim, bias=False)

        if noisy_gating:
            # noise scale conditioned on projected text
            self.noise_network = nn.Linear(e_dim, num_latent_tokens)

    def compute_load_balancing_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Switch-style batch-level load balancing:
        encourage uniform average usage of experts.
        gate_probs: [B, E]
        """
        expert_usage = gate_probs.mean(dim=0)                     # [E]
        target = torch.full_like(expert_usage, 1.0 / gate_probs.size(1))
        return F.mse_loss(expert_usage, target)

    def forward(self, text_emb: torch.Tensor, q_video_emb: torch.Tensor, return_token_idx=False):
        """
        Enhanced forward pass with soft mixture of experts.
        Args:
            text_emb:     [B, text_dim] - Input text embeddings
            q_video_emb:  [B, E, e_dim] - Quantized video tokens
            return_token_idx: bool - Whether to return argmax token indices for compatibility
        Returns:
            soft_mixture: [B, e_dim] - Weighted mixture of all tokens (actual routed embedding)
            gate_probs:   [B, E] - Gate probabilities representing mixture weights
            top1_idx:     [B] (optional) - Argmax indices for semantic ID compatibility
        """
        B, E, Dq = q_video_emb.shape
        assert E == self.num_latent_tokens, f"expected {self.num_latent_tokens}, got {E}"
        assert Dq == self.e_dim, f"q_video_emb last dim {Dq} must equal e_dim {self.e_dim}"

        # 1) Text: detach to avoid backprop into GT text tower
        t = self.text_norm(text_emb)        # [B, text_dim]
        t = self.t_proj(t)                  # [B, e_dim]
        t = F.normalize(t, p=2, dim=-1)     # L2 normalize

        # 2) Tokens already in e_dim: normalize per-expert for cosine
        q = F.normalize(q_video_emb, p=2, dim=-1)    # [B, E, e_dim]

        # 3) Cosine similarity logits: <t, q_e> / T
        # logits[b, e] = sum_d t[b, d] * q[b, e, d]
        logits = torch.einsum('bd,bed->be', t, q) / self.temperature   # [B, E]

        # 4) Apply soft temperature for mixture control
        gate_logits = logits / self.soft_temperature  # [B, E]

        # 5) Noisy gating (training only) - applied to temperature-scaled logits
        if self.noisy_gating and self.training:
            noise_std = F.softplus(self.noise_network(t)) + self.noise_epsilon  # [B, E]
            gate_logits = gate_logits + torch.randn_like(gate_logits) * noise_std

        # 6) Gate probabilities in full precision for numerical stability
        gate_probs = F.softmax(gate_logits.float(), dim=-1).to(gate_logits.dtype)  # [B, E]

        # 7) Soft weighted mixture using torch.einsum
        # gate_probs: [B, E], q_video_emb: [B, E, e_dim] -> [B, e_dim]
        soft_mixture = torch.einsum('be,bed->bd', gate_probs, q_video_emb)

        # 8) Compatibility: Provide argmax indices for semantic ID tools
        if return_token_idx:
            top1_idx = torch.argmax(gate_probs, dim=-1)  # [B]
            return soft_mixture, gate_probs, top1_idx

        return soft_mixture, gate_probs


class WeightTiedLMHead(nn.Module):
    def __init__(self, embeddings, target_codebook_size):
        super().__init__()
        self.weight = embeddings.weight
        self.target_codebook_size = target_codebook_size

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # Get the weights for the target codebook size
        weight = self.weight[:self.target_codebook_size]  # Shape: [target_codebook_size, embed_dim]
        # Compute the logits by matrix multiplication
        logits = torch.matmul(x, weight.t())  # Shape: [batch_size, seq_len, target_codebook_size]
        return logits


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim
        
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]
        
        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias
        
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)
        
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class AttentiveBlock(nn.Module):
    
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()
        
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        
        return x


class AttentionPoolingBlock(AttentiveBlock):
    
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x