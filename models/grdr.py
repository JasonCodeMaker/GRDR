from abc import ABC
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, Tensor
from tqdm import tqdm
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import ModelOutput

from utils.model_utils import sinkhorn_raw


@dataclass
class QuantizeOutput(ModelOutput):
    """Output from query path (text -> T5 encoder -> code projection)."""
    logits: Optional[torch.FloatTensor] = None
    continuous_embeds: Optional[torch.FloatTensor] = None
    total_embeds: Optional[torch.FloatTensor] = None
    quantized_embeds: Optional[torch.FloatTensor] = None
    discrete_codes: Optional[torch.LongTensor] = None
    probability: Optional[torch.FloatTensor] = None
    code_logits: Optional[torch.FloatTensor] = None


@dataclass
class VideoOutput(ModelOutput):
    """Output from Video path (video features -> VideoRQVAE)."""
    logits: Optional[torch.FloatTensor] = None
    continuous_embeds: Optional[torch.FloatTensor] = None
    quantized_embeds: Optional[torch.FloatTensor] = None
    reconstructed_features: Optional[torch.FloatTensor] = None
    discrete_codes: Optional[torch.LongTensor] = None
    probability: Optional[torch.FloatTensor] = None
    code_logits: Optional[torch.FloatTensor] = None
    rq_loss: Optional[torch.FloatTensor] = None


class Codebook(nn.Module):
    """Codebook for computing cosine similarity between hidden states and embeddings."""

    def __init__(self, codebook_embedding):
        super().__init__()
        self.codebook = codebook_embedding

    def forward(self, hidden):
        # Normalize both hidden states and codebook weights to use cosine similarity
        # This aligns T5's probability calculation with VideoRQVAE's normalized distance computation
        # When both vectors are unit-normalized, dot product = cosine similarity
        hidden_normalized = F.normalize(hidden, dim=-1, eps=1e-12)
        weight_normalized = F.normalize(self.codebook.weight, dim=-1, eps=1e-12)

        # Compute cosine similarity via dot product of normalized vectors
        # logits = ⟨hidden_norm, weight_norm⟩ = cos(θ)
        # This matches VideoRQVAE's scoring: argmax(cos(θ)) = argmin(2 - 2cos(θ)) = argmin(distance)
        logits = F.linear(hidden_normalized, weight_normalized, bias=None)

        return logits


class GRDR(nn.Module, GenerationMixin, ABC):
    """
    GRDR (Generative Retrieval with Dual-path Representation) Model.

    This model implements dual-path forward:
    - Query path (input_ids provided): Use T5 encoder -> project to code space
    - Video path (video_features provided): Use VideoRQVAE encoder-quantizer
    """
    # Required attributes for GenerationMixin compatibility
    base_model_prefix = ""
    _supports_cache_class = False

    def __init__(self, model, use_constraint: bool, sk_epsilon: float = 0.03, sk_iters: int = 100, code_length=1,
                 zero_inp=False, code_number=10, videorqvae=None):
        super().__init__()
        self.model = model
        self.config = model.config
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
        self.get_encoder = model.get_encoder
        self.device = model.device
        self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self.can_generate = lambda: True
        hidden_size = model.config.hidden_size

        # Force use_constraint to False to ensure consistent behavior between T5 and VideoRQVAE
        self.use_constraint, self.sk_epsilon, self.sk_iters = False, sk_epsilon, sk_iters

        self.video_rqvae = videorqvae

        self.code_embedding = nn.ModuleList([
            videorqvae.rq.vq_layers[i].embedding for i in range(code_length)
        ])

        self.centroids = nn.ModuleList([
            Codebook(self.code_embedding[i]) for i in range(code_length)
        ])

        # Separate start token embedding (not shared with any codebook)
        # This prevents collision with code=0 samples in K-means
        self.start_token_embedding = nn.Parameter(
            torch.randn(1, hidden_size) * 0.02  # Small random initialization
        )

        self.logit_scale = nn.Parameter(torch.tensor(4.605))

        self.code_length = code_length
        self.zero_inp = zero_inp
        self.code_number = code_number

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):
        return {"decoder_input_ids": input_ids, "encoder_outputs": encoder_outputs, "attention_mask": attention_mask}

    @torch.no_grad()
    def quantize(self, probability, use_constraint=None):
        distances = -probability
        use_constraint = self.use_constraint if use_constraint is None else use_constraint
        if not use_constraint:
            codes = torch.argmin(distances, dim=-1)
        else:
            distances = self.center_distance_for_constraint(distances)
            distances = distances.double()
            Q = sinkhorn_raw(
                -distances,
                self.sk_epsilon,
                self.sk_iters,
                use_distrib_train=dist.is_initialized()
            )
            codes = torch.argmax(Q, dim=-1)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
        return codes

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        if dist.is_initialized():
            dist.all_reduce(max_distance, torch.distributed.ReduceOp.MAX)
            dist.all_reduce(min_distance, torch.distributed.ReduceOp.MIN)
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert torch.all(amplitude > 0)
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, aux_ids=None,
                video_features=None, token_idx=None, return_code=False, return_quantized_embedding=False,
                use_constraint=None, encoder_outputs=None, return_residual_layer=None, return_all=False, **kwargs):
        """
        Dual-path forward:
        - Query path (input_ids provided): Use T5 encoder -> project to code space
        - Video path (video_features provided): Use VideoRQVAE encoder-quantizer
        """
        use_constraint = use_constraint if use_constraint is not None else self.use_constraint

        # Video path: video features -> VideoRQVAE
        if video_features is not None:
            return self._forward_video(video_features, token_idx, use_constraint, return_quantized_embedding, return_residual_layer, return_all)

        # Query path: text tokens -> T5 encoder -> code projection
        # Also handles encoder_outputs (from beam search subsequent steps)
        if input_ids is not None or encoder_outputs is not None:
            return self._forward_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_constraint=use_constraint,
                decoder_input_ids=decoder_input_ids,
                aux_ids=aux_ids,
                return_code=return_code,
                return_quantized_embedding=return_quantized_embedding,
                encoder_outputs=encoder_outputs
            )

        raise ValueError("Either input_ids or video_features must be provided")

    def _forward_video(self, video_features, token_idx, use_constraint, return_quantized_embedding, return_residual_layer=None, return_all=False):
        # Encode video features into latent tokens
        video_encoded_features = self.video_rqvae.encoder(video_features)

        # Quantize ALL latent tokens and get raw distances
        video_quantized_features, rq_loss, indices, distances, residuals = self.video_rqvae.rq(
            video_encoded_features, use_sk=use_constraint, return_probs=True
        )

        # If return_all, skip token selection and return all latent tokens
        if return_all:
            video_decoded_features, reconstructed_features = self.video_rqvae.decoder(video_quantized_features)

            return VideoOutput(
                logits=None,
                continuous_embeds=video_encoded_features,
                quantized_embeds=video_quantized_features,
                reconstructed_features=reconstructed_features,
                discrete_codes=None,
                probability=None,
                code_logits=None,
                rq_loss=rq_loss,
            )

        # Select the token assigned by k-means for each sample
        B = video_encoded_features.size(0)
        batch_indices = torch.arange(B, device=video_encoded_features.device)

        # Compute residuals up to specified layer if needed (for K-Means initialization)
        if return_residual_layer is not None and return_residual_layer > 0:
            selected_encoded_features = residuals[return_residual_layer-1][batch_indices, token_idx]
        else:
            selected_encoded_features = video_encoded_features[batch_indices, token_idx]

        selected_quantized_features = video_quantized_features[batch_indices, token_idx]
        selected_indices = indices[batch_indices, token_idx]

        # Convert raw distances to cosine similarity for the selected token
        scale = self.logit_scale.exp().clamp(min=20.0, max=100.0)

        selected_code_logits = []
        for layer_distances in distances:
            layer_logits = 1 - layer_distances[batch_indices, token_idx] / 2
            layer_logits = layer_logits * scale
            selected_code_logits.append(layer_logits)
        code_logits = torch.stack(selected_code_logits, dim=1)

        probability = code_logits[:, -1].contiguous()
        discrete_codes = selected_indices[:, -1].contiguous()

        if code_logits.size(1) == 1:
            return_code_logits = None
        else:
            return_code_logits = code_logits[:, :-1].contiguous()

        video_decoded_features, reconstructed_features = self.video_rqvae.decoder(video_quantized_features)

        return VideoOutput(
            logits=code_logits,
            continuous_embeds=selected_encoded_features,
            quantized_embeds=selected_quantized_features,
            reconstructed_features=reconstructed_features,
            discrete_codes=discrete_codes,
            probability=probability,
            code_logits=return_code_logits,
            rq_loss=rq_loss,
        )

    def _forward_text(self, input_ids=None, attention_mask=None, decoder_input_ids=None, aux_ids=None, return_code=False,
                return_quantized_embedding=False, use_constraint=None, encoder_outputs=None, **kwargs):
        # Get batch size and device from available tensors
        if input_ids is not None:
            batch_size, device = input_ids.size(0), input_ids.device
        elif decoder_input_ids is not None:
            batch_size, device = decoder_input_ids.size(0), decoder_input_ids.device
        else:
            batch_size = encoder_outputs[0].size(0)
            device = encoder_outputs[0].device

        if decoder_input_ids is None or self.zero_inp:
            decoder_input_ids = torch.zeros(batch_size, self.code_length).long().to(device)

        decoder_inputs_embeds = []
        batch_size = decoder_input_ids.size(0)
        for i in range(min(decoder_input_ids.size(1), len(self.code_embedding))):
            if i == 0:
                decoder_inputs_embeds.append(self.start_token_embedding.expand(batch_size, -1))
            else:
                code_embedding = self.code_embedding[i - 1]
                decoder_inputs_embeds.append(code_embedding(decoder_input_ids[:, i]))
        decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs
        )
        decoder_outputs = model_outputs.decoder_hidden_states[-1]

        dense_embed = decoder_outputs[:, -1].contiguous()
        total_embeds = decoder_outputs.sum(dim=1).contiguous()

        scale = self.logit_scale.exp().clamp(min=20.0, max=100.0)
        code_logits = []
        for i in range(min(decoder_input_ids.size(1), len(self.code_embedding))):
            centroid = self.centroids[i]
            code_logits.append(centroid(decoder_outputs[:, i]) * scale)
        code_logits = torch.stack(code_logits, dim=1)

        probability = code_logits[:, -1].contiguous()
        discrete_codes = self.quantize(probability, use_constraint=use_constraint)

        if aux_ids is None:
            aux_ids = discrete_codes

        if self.code_length == 1:
            return_code_logits = None
        else:
            return_code_logits = code_logits[:, :-1].contiguous()

        quant_output = QuantizeOutput(
            logits=code_logits,
            continuous_embeds=dense_embed,
            total_embeds=total_embeds,
            quantized_embeds=None,
            discrete_codes=discrete_codes,
            probability=probability,
            code_logits=return_code_logits,
        )
        return quant_output

    @torch.no_grad()
    def gen_sid(self, data_loader, return_quantized_features=False):
        """Generate semantic IDs for videos in the data loader."""
        sample_codes_dict = {}
        sid_to_features = {} if return_quantized_features else None

        for batch in tqdm(data_loader):
            video_features = batch['video_features'].cuda()

            if return_quantized_features:
                indices, q_features = self.video_rqvae.get_indices(video_features, return_quantized_features=True)
            else:
                indices = self.video_rqvae.get_indices(video_features)

            for i in range(len(batch['video_ids'])):
                video_id = batch['video_ids'][i]
                code = indices[i].cpu().tolist()
                sample_codes_dict[video_id] = code

                if return_quantized_features:
                    for token_idx, token_code in enumerate(code):
                        sid_str = str([0, *token_code])
                        if sid_str not in sid_to_features:
                            sid_to_features[sid_str] = q_features[i, token_idx].cpu().numpy()

        if return_quantized_features:
            return sample_codes_dict, sid_to_features
        return sample_codes_dict