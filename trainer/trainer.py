import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import wandb
from torch import Tensor
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, get_constant_schedule

from models.t5 import T5ForConditionalGeneration
from transformers import T5Config

from models.grdr import GRDR, QuantizeOutput, VideoOutput
from utils.model_utils import create_videorqvae, get_optimizer, CodeDriftMonitor
from utils.training_utils import safe_load, safe_load_embedding, safe_save
from utils.data_utils import load_shared_features
from data.video_dataset import VideoTextDataset, collate_fn


class OurTrainer:
    """Trainer class with static methods for training and testing steps."""

    @staticmethod
    def _gather_tensor(t: Tensor, local_rank):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[local_rank] = t
        return all_tensors

    @staticmethod
    def gather_tensors(t: Tensor, local_rank=None):
        if local_rank is None:
            local_rank = dist.get_rank()
        return torch.cat(OurTrainer._gather_tensor(t, local_rank))

    @staticmethod
    @torch.no_grad()
    def test_step(model: GRDR, batch, use_constraint=None):
        """
        Test step compatible with VideoTextDataset batch structure.

        Expected batch keys:
        - caption_tokens: [B, seq_len] tokenized captions (query)
        - attention_mask: [B, seq_len] attention mask for captions
        - video_features: [B, 512] video embeddings (Video)
        - token_idx: [B] k-means assigned token indices
        """
        # Query path: text caption -> T5 encoder
        query_outputs: QuantizeOutput = model(
            input_ids=batch['caption_tokens'],
            attention_mask=batch['attention_mask'],
            return_code=False,
            return_quantized_embedding=False,
            use_constraint=use_constraint
        )

        # Video path: video features -> VideoRQVAE with token selection
        vid_outputs: VideoOutput = model(
            video_features=batch['video_features'],
            token_idx=batch['token_idx'],
            return_code=False,
            return_quantized_embedding=False,
            use_constraint=use_constraint
        )

        return query_outputs, vid_outputs

    @staticmethod
    def simple_train_step(model: GRDR, batch, gathered=None):
        """
        Simple train step compatible with VideoTextDataset batch structure.
        Uses only commitment loss on query probability.
        """
        # Query path: text caption -> T5 encoder
        query_outputs: QuantizeOutput = model(
            input_ids=batch['caption_tokens'],
            attention_mask=batch['attention_mask']
        )

        # Video path: video features -> VideoRQVAE with token selection
        vid_outputs: QuantizeOutput = model(
            video_features=batch['video_features'],
            token_idx=batch['token_idx']
        )

        # Commitment loss
        query_prob = query_outputs.probability
        codes_vid = vid_outputs.discrete_codes
        query_ce_loss = F.cross_entropy(query_prob, codes_vid.detach())

        return dict(
            loss=query_ce_loss,
        )

    @staticmethod
    def train_step(model: GRDR, batch, current_layer, gathered=None):
        """
        Train step compatible with VideoTextDataset batch structure.

        Expected batch keys:
        - caption_tokens: [B, seq_len] tokenized captions (query)
        - attention_mask: [B, seq_len] attention mask for captions
        - video_features: [B, 512] video embeddings (Video)
        - video_ids: List[str] unique video identifiers (e.g., video7015_0)
        - token_idx: [B] k-means assigned token indices
        """
        # Video path: video features -> VideoRQVAE with token selection
        video_features = batch['video_features']
        vid_outputs: VideoOutput = model(
            video_features=video_features,
            token_idx=batch['token_idx'],
            decoder_input_ids=batch['ids'],
            aux_ids=batch['aux_ids'],
            return_code=True,
            return_quantized_embedding=True,
            return_residual_layer= model.code_length if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.module.code_length - 1
        )

        # Query path: text caption -> T5 encoder
        query_outputs: QuantizeOutput = model(
            input_ids=batch['caption_tokens'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=batch['ids'],
            aux_ids=batch['aux_ids'],
            return_code=True,
            return_quantized_embedding=True
        )

        # Extract embeddings
        query_embeds = query_outputs.total_embeds
        vid_embeds = vid_outputs.continuous_embeds
        recon_video_features = vid_outputs.reconstructed_features
        codes_vid = vid_outputs.discrete_codes
        query_prob = query_outputs.probability
        vid_prob = vid_outputs.probability

        if gathered is None:
            gathered = dist.is_initialized()

        # Contrastive loss: query vs doc
        cl_loss = OurTrainer.compute_contrastive_loss(query_embeds, vid_embeds, gathered=False)

        # Video Reconstruction Loss
        L2_video_features = F.normalize(video_features.detach(), p=2, dim=-1, eps=1e-12)
        L2_recon_video_features = F.normalize(recon_video_features, p=2, dim=-1, eps=1e-12)
        cl_dd_loss = (1 - F.cosine_similarity(L2_video_features, L2_recon_video_features, dim=-1)).mean()

        # Commitment loss: query should predict video sID codes
        query_ce_loss = F.cross_entropy(query_prob, codes_vid.detach(), label_smoothing=0.1)

        ce_loss = query_ce_loss

        # Code prediction loss (multi-layer)
        code_loss = 0
        if query_outputs.code_logits is not None and vid_outputs.code_logits is not None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                code_number = model.module.code_number
            else:
                code_number = model.code_number

            prior_codes = batch['ids'][:, 1:].contiguous().view(-1)
            query_code_loss = F.cross_entropy(
                query_outputs.code_logits.view(-1, code_number),
                prior_codes
            )

            # Video-side code loss (prevent code drift during progressive training)
            vid_code_loss = F.cross_entropy(
                vid_outputs.code_logits.view(-1, code_number),
                prior_codes
            )

            code_loss = query_code_loss + vid_code_loss

        # VideoRQVAE quantization loss
        rq_loss = vid_outputs.rq_loss if vid_outputs.rq_loss is not None else 0

        return dict(
            cl_loss=cl_loss,
            ce_loss=ce_loss,
            code_loss=code_loss,
            cl_dd_loss=cl_dd_loss,
            rq_loss=rq_loss
        )

    @staticmethod
    def compute_contrastive_loss(query_embeds, vid_embeds, gathered=True):
        if gathered:
            gathered_query_embeds = OurTrainer.gather_tensors(query_embeds)
            gathered_doc_embeds = OurTrainer.gather_tensors(vid_embeds)
        else:
            gathered_query_embeds = query_embeds
            gathered_doc_embeds = vid_embeds
        effective_bsz = gathered_query_embeds.size(0)
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(gathered_query_embeds, gathered_doc_embeds.transpose(0, 1))
        co_loss = F.cross_entropy(similarities, labels)
        return co_loss


def build_loss_weights(config, phase):
    """
    Build loss weight dictionary from config for a specific training phase.

    Args:
        config: Configuration dictionary with loss weight parameters
        phase: Training phase (1=pre-train, 2=main, 3=fit)

    Returns:
        Dictionary of loss weights for the specified phase
    """
    prefix = f'w{phase}_'
    return {
        'cl_loss': config.get(f'{prefix}cl_loss', 0),
        'ce_loss': config.get(f'{prefix}ce_loss', 0),
        'code_loss': config.get(f'{prefix}code_loss', 0),
        'cl_dd_loss': config.get(f'{prefix}cl_dd_loss', 0),
        'rq_loss': config.get(f'{prefix}rq_loss', 0)
    }


def train(config, global_step=0):
    """Main training function for GRDR model."""
    # Import here to avoid circular imports
    from trainer.evaluator import eval_retrieval, save_code

    accelerator = Accelerator(gradient_accumulation_steps=1)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    model_name = config.get('model_name', 't5-base')
    code_num = config.get('code_num', 128)
    code_length = config.get('code_length', 4)
    prev_model = config.get('prev_model', None)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size')
    lr = config.get('lr')
    end_epoch = epochs

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    accelerator.print(save_path)

    # Load T5 model with explicit config and dtype control
    t5_config = T5Config.from_pretrained(model_name)
    t5_config.dropout_rate = config.get('dropout_rate', t5_config.dropout_rate)

    # Determine torch dtype based on config
    if config.get('float16', False):
        torch_dtype = torch.float16
    elif config.get('bf16', False):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    videorqvae = create_videorqvae(
        code_num=code_num,
        code_length=code_length,
        num_latent_tokens=config.get('num_latent_tokens', 4),
        e_dim=t5_config.d_model,
        in_dim=config.get('in_dim', 512),
        device=accelerator.device
    )
    accelerator.print(f'VideoRQVAE created with e_dim={t5_config.d_model} (matching T5 hidden_size)')

    videorqvae = videorqvae.to(accelerator.device)

    t5 = T5ForConditionalGeneration.from_pretrained(model_name,
                                                    torch_dtype=torch_dtype,
                                                    config=t5_config)
    model = GRDR(model=t5, code_length=code_length,
                 use_constraint=False, sk_epsilon=1, zero_inp=False, code_number=code_num,
                 videorqvae=videorqvae)

    if prev_model is not None:
        safe_load(model.video_rqvae, f'{prev_model}.videorqvae')
        safe_load(model.model, f'{prev_model}.model')
        safe_load(model.centroids, f'{prev_model}.centroids')
        safe_load_embedding(model.code_embedding, f'{prev_model}.embedding')
        # Load start token embedding if available (backward compatible with old checkpoints)
        start_token_path = f'{prev_model}.start_token'
        if os.path.exists(start_token_path):
            model.start_token_embedding.data = torch.load(start_token_path, map_location='cpu')
            accelerator.print(f'Loaded start_token_embedding from {start_token_path}')

    # Print total model parameters in millions
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f'Total model parameters: {total_params / 1e6:.2f}M')

    if config.get('codebook_init', None) is not None:
        from run import read_pkl
        model.code_embedding[-1].weight.data = torch.tensor(read_pkl(config.get('codebook_init')))

    # Phase-specific freezing:
    if config.get('loss_w') == 3:
        for embedding in model.code_embedding:
            embedding.requires_grad_(False)
        for param in model.video_rqvae.parameters():
            param.requires_grad_(False)
    else:
        # Since centroids reference code_embedding, freezing code_embedding freezes centroids too
        for i in range(code_length - 1):
            model.code_embedding[i].requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    dataset_name = config.get('dataset', 'msrvtt')
    features_root = config.get('features_root', './data_process/datasets/features')
    use_pseudo_queries = config.get('use_pseudo_queries', False)

    accelerator.print(f'Loading features for {dataset_name}...')
    feature_cache = load_shared_features(
        dataset_name=dataset_name,
        features_root=features_root,
        logger=accelerator,
        use_pseudo_queries=use_pseudo_queries
    )

    num_latent_tokens = config.get('num_latent_tokens', 4)
    cache_dir = config.get('cache_dir', './cache')

    video_codes, aux_ids = None, None

    # Build code dict
    if prev_id is not None:
        accelerator.print(f'Loading previous codes from {prev_id}')
        prev_codes_dict = json.load(open(prev_id))
        video_codes = {k: [0, *v] for k, v in prev_codes_dict.items()}
        accelerator.print(f'Loaded {len(video_codes)} hierarchical codes')

    # Create VideoTextDataset with codes
    dataset = VideoTextDataset(
        dataset_name=dataset_name,
        video_features=feature_cache['train_video'],
        text_features=feature_cache['train_text'],
        tokenizer=tokenizer,
        split='train',
        max_text_len=128,
        num_latent_tokens=num_latent_tokens,
        cache_dir=cache_dir,
        ids=video_codes,
        aux_ids=aux_ids,
        use_pseudo_queries=use_pseudo_queries
    )
    accelerator.print(f'Dataset size: {len(dataset)} video-caption pairs with hierarchical codes')

    # Create validation dataset from train_dataset attributes (for validation evaluation)
    val_dataset = VideoTextDataset(
        dataset_name=dataset_name,
        video_features=feature_cache['train_video'],
        text_features=feature_cache['train_text'],
        tokenizer=tokenizer,
        split='val',
        max_text_len=128,
        num_latent_tokens=num_latent_tokens,
        cache_dir=cache_dir,
        ids=video_codes,
        use_pseudo_queries=use_pseudo_queries
    )
    accelerator.print(f'Val dataset size: {len(val_dataset)} video-caption pairs')

    # Create test dataset for evaluation
    test_dataset = VideoTextDataset(
        dataset_name=dataset_name,
        video_features=feature_cache['test_video'],
        text_features=feature_cache['test_text'],
        tokenizer=tokenizer,
        split='test',
        max_text_len=128,
        num_latent_tokens=num_latent_tokens,
        cache_dir=cache_dir,
        ids=None
    )
    accelerator.print(f'Test dataset size: {len(test_dataset)} video-caption pairs')

    # Create collate function with tokenizer closure
    collate_wrapper = lambda batch: collate_fn(batch, tokenizer, max_length=128)

    # Create seeded generator for reproducibility
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_wrapper,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        generator=g,
        worker_init_fn=lambda worker_id: np.random.seed(config.get('seed', 42) + worker_id)
    )
    # Calculate encoder LR scale based on loop index
    loop = config.get('loop', 0)
    encoder_lr_scale = 1.0 ** loop
    accelerator.print(f'Loop {loop}: Encoder LR scale = {encoder_lr_scale:.4f} (base_lr={lr}, encoder_lr={lr * encoder_lr_scale:.2e})')

    # Use get_optimizer to apply custom learning rates
    optimizer = get_optimizer(model, lr=lr, code_length=code_length, encoder_lr_scale=encoder_lr_scale)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    scheduler = get_constant_schedule(optimizer)

    # Build loss weights from config based on training phase
    loss_w = build_loss_weights(config, config['loss_w'])

    step, epoch = 0, 0
    epoch_step = len(data_loader)
    last_checkpoint = None
    best_metric = 0.0

    # Initialize code drift monitor for loop > 0 to track if previous layer codes drift
    drift_monitor = None
    if loop > 0 and prev_id is not None:
        drift_monitor = CodeDriftMonitor(num_layers=loop, token_idx_mapping=dataset.text_groups)
        drift_monitor.current_layer = loop
        baseline_codes = json.load(open(prev_id))
        num_samples = drift_monitor.snapshot_baseline(baseline_codes)
        accelerator.print(f'[Drift Monitor] Loaded {num_samples} baseline codes from {prev_id}')

    for _ in range(epochs):
        accelerator.print(f'Training {save_path} {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            step += 1
            global_step += 1
            with accelerator.accumulate(model):
                losses = OurTrainer.train_step(model, batch, current_layer=loop, gathered=False)
                loss = sum([v * loss_w[k] for k, v in losses.items()])
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss = accelerator.gather(loss).mean().item()
                loss_report.append(loss)
                tk0.set_postfix(loss=sum(loss_report[-100:]) / len(loss_report[-100:]))

                # Log losses to wandb
                if accelerator.is_main_process:
                    wandb_log = {f"train/{k}": (v.item() if isinstance(v, torch.Tensor) else v) if loss_w.get(k, 0) != 0 else 0 for k, v in losses.items()}
                    wandb_log["train/total_loss"] = loss
                    wandb.log(wandb_log, step=global_step)

        # Evaluation at end of epoch
        is_pretrain = (loss_w.get('ce_loss', 0) == 0)
        _, current_metric = eval_retrieval(model, dataset, val_dataset, test_dataset, tokenizer, batch_size, accelerator, global_step,
                                           is_pretrain=is_pretrain, code_length=code_length, drift_monitor=drift_monitor)
        best_metric, last_checkpoint, is_new_best = safe_save(accelerator, model, save_path, best_metric, current_metric,
                                                              last_checkpoint=last_checkpoint)
        if is_new_best:
            save_code(model, dataset, video_codes, tokenizer, batch_size, save_path)
        epoch += 1
        model.train()

    return last_checkpoint, global_step
