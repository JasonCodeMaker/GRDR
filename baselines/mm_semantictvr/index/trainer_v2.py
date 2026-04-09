import logging
import json

import numpy as np
import torch
from time import time
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from typing import List, NamedTuple, Optional, Tuple
from .utils import ensure_dir,set_color,get_local_time
import os
from transformers import get_scheduler


class CollisionMetrics(NamedTuple):
    inter_collision_rate: float
    total_collision_rate: float
    inter_duplicate_max: int
    total_freq_max: int

class Trainer(object):

    def __init__(self, args, model, video_key=None, model_type=None):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.vq_lr = args.vq_lr
        self.lr_scheduler_type = args.lr_scheduler
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = torch.device(f"cuda:{args.device}")
        self.ckpt_dir = args.ckpt_dir
        # saved_model_dir = "{}".format(get_local_time())
        # self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.train_log_path = os.path.join(self.ckpt_dir, "train.log")
        
        # Setup file logging to capture all output
        self._setup_file_logging()

        # Learning rate scheduler parameters
        self.lr_scheduler_type = args.lr_scheduler
        self.warmup_ratio = args.warmup_ratio
        self.min_lr_ratio = args.min_lr_ratio

        self.best_loss = np.inf
        self.best_test_loss = np.inf
        self.best_recall_at_1 = 0.0
        self.best_recall_at_5 = 0.0
        self.best_recall_at_10 = 0.0
        self.best_test_recon_loss_ckpt = "best_test_loss_model.pth"
        self.best_recall_at_1_ckpt = "best_recall_at_1_model.pth"
        self.best_recall_at_5_ckpt = "best_recall_at_5_model.pth"
        self.best_recall_at_10_ckpt = "best_recall_at_10_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = None  # Will be initialized in fit() when we know the data loader size
        self.model = self.model.to(self.device)

        # Text guidance parameters
        self.use_wandb = args.use_wandb
        self.text_guided = args.text_guided
        self.multi_text_mode = args.multi_text_mode

        # Contrastive learning weight (training hyperparameter, not model parameter)
        self.contrastive_loss_weight = getattr(args, 'contrastive_loss_weight', 0.0)

        self.video_key = video_key
        self.model_type = model_type

    def _setup_file_logging(self):
        """Setup file handler to save all logs to train.log"""
        root_logger = logging.getLogger()
        
        # Check if file handler for this path already exists
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if hasattr(handler, 'baseFilename') and handler.baseFilename == os.path.abspath(self.train_log_path):
                    self.logger.info(f"File logging already configured: {self.train_log_path}")
                    return
        
        # Create file handler
        file_handler = logging.FileHandler(self.train_log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger to capture all logs
        root_logger.addHandler(file_handler)
        
        self.logger.info(f"Logging to file: {self.train_log_path}")

    def _build_optimizer(self):
        # Separate VectorQuantizer parameters from other parameters
        vq_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Check if this is a VectorQuantizer parameter
                if ('rq.vq_layers.' in name and
                    ('embedding.weight' in name or 'codebook_projection.' in name)):
                    vq_params.append(param)
                else:
                    other_params.append(param)

        # Log parameter group information
        self.logger.info(f"VectorQuantizer parameters: {len(vq_params)} (lr={self.vq_lr})")
        self.logger.info(f"Other parameters: {len(other_params)} (lr={self.lr})")

        # Create parameter groups
        param_groups = [
            {'params': other_params, 'lr': self.lr},
            {'params': vq_params, 'lr': self.vq_lr}
        ]

        learner = self.learner
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(param_groups, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(param_groups, weight_decay=weight_decay)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(param_groups, weight_decay=weight_decay)
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(param_groups)
        return optimizer

    def _build_scheduler(self, data_loader):
        """Build learning rate scheduler using transformers library."""
        if self.lr_scheduler_type == 'constant':
            return None

        # Calculate total training steps
        total_steps = self.epochs * len(data_loader)
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.logger.info(f"Learning rate scheduler: {self.lr_scheduler_type}")
        self.logger.info(f"Total training steps: {total_steps}")
        self.logger.info(f"Warmup steps: {warmup_steps} ({self.warmup_ratio:.1%})")
        if self.min_lr_ratio > 0:
            self.logger.info(f"Min LR ratio: {self.min_lr_ratio} (Note: transformers cosine scheduler uses 0)")

        # Map our scheduler types to transformers scheduler names
        scheduler_map = {
            'cosine': 'cosine',
            'linear': 'linear',
        }

        scheduler_name = scheduler_map.get(self.lr_scheduler_type)

        # Create scheduler using transformers library
        scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.logger.info(f"Successfully created {scheduler_name} scheduler from transformers")
        self.logger.info(f"Scheduler handles multi-parameter groups: VQ lr={self.vq_lr:.6f}, main lr={self.lr:.6f}")
        return scheduler

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _extract_video_tensor(self, batch):
        """Retrieve the video tensor from a dataloader batch in eval routines."""
        if isinstance(batch, dict):
            if hasattr(self, 'video_key') and self.video_key in batch:
                video_tensor = batch[self.video_key]
            else:
                tensor_values = [v for v in batch.values() if torch.is_tensor(v)]
                if not tensor_values:
                    raise ValueError("No tensor-valued entries found in batch for collision evaluation")
                video_tensor = tensor_values[0]
        else:
            video_tensor = batch

        if not torch.is_tensor(video_tensor):
            raise TypeError(f"Expected tensor for video data, got {type(video_tensor)}")

        return video_tensor.to(self.device)

    def _accumulate_collision_counts(self, indices, token_counts):
        """Accumulate token statistics and intra-video duplicate counts.

        Returns the token count per video and duplicate counts per video so the
        caller can aggregate collision metrics according to the current
        definitions.
        """
        indices_cpu = indices.detach().cpu()

        if indices_cpu.dim() == 1:
            indices_cpu = indices_cpu.unsqueeze(0).unsqueeze(0)
        elif indices_cpu.dim() == 2:
            indices_cpu = indices_cpu.unsqueeze(1)

        if indices_cpu.dim() != 3:
            raise ValueError(f"Unexpected indices shape {tuple(indices_cpu.shape)}; expected B x T x L")

        tokens_per_video: List[int] = []
        duplicates_per_video: List[int] = []

        for video_codes in indices_cpu.tolist():
            token_tuples = []
            for token_codes in video_codes:
                token_tuple = tuple(int(idx) for idx in token_codes)
                token_counts[token_tuple] += 1
                token_tuples.append(token_tuple)

            num_tokens = len(token_tuples)
            unique_tokens = len(set(token_tuples))

            tokens_per_video.append(num_tokens)
            duplicates_per_video.append(max(0, num_tokens - unique_tokens))

        return tokens_per_video, duplicates_per_video

    def _finalize_collision_metrics(self, total_tokens, inter_duplicates, token_counts):
        """Compute collision metrics from accumulated frequency counters."""
        if total_tokens == 0:
            return CollisionMetrics(0.0, 0.0, 0, 0)

        total_inter_duplicates = sum(inter_duplicates)
        inter_collision = total_inter_duplicates / total_tokens
        inter_max = max(inter_duplicates) if inter_duplicates else 0

        total_unique = len(token_counts)
        total_collision = (total_tokens - total_unique) / total_tokens
        total_max = max(token_counts.values()) if token_counts else 0

        return CollisionMetrics(
            inter_collision_rate=inter_collision,
            total_collision_rate=total_collision,
            inter_duplicate_max=inter_max,
            total_freq_max=total_max,
        )

    @torch.no_grad()
    def _valid_epoch(self, valid_data_loader):
        """Validation epoch for collision metrics tracking.
        VideoRQVAE_V2 specific implementation.
        """
        self.model.eval()

        iter_data =tqdm(
                valid_data_loader,
                total=len(valid_data_loader),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        token_counts = defaultdict(int)
        total_tokens = 0
        inter_duplicate_counts: List[int] = []

        for batch_idx, batch in enumerate(iter_data):
            video_batch = self._extract_video_tensor(batch)

            # Get indices for collision tracking
            indices, _ = self.model.get_indices(video_batch)

            # Collision metrics computation
            tokens_per_video, duplicates_per_video = self._accumulate_collision_counts(indices, token_counts)
            total_tokens += sum(tokens_per_video)
            inter_duplicate_counts.extend(duplicates_per_video)

        collision_metrics = self._finalize_collision_metrics(total_tokens, inter_duplicate_counts, token_counts)

        return collision_metrics

    @torch.no_grad()
    def _test_epoch(self, test_data_loader):
        """Test epoch with reconstruction loss measurement.
        VideoRQVAE_V2 specific implementation.
        """
        self.model.eval()

        iter_data = tqdm(
            test_data_loader,
            total=len(test_data_loader),
            ncols=100,
            desc=set_color(f"Test   ", "pink"),
        )

        # Measure Reconstruction Loss for test data
        recon_loss_list = []
        text_recon_loss_list = []
        token_counts = defaultdict(int)
        total_tokens = 0
        inter_duplicate_counts: List[int] = []

        # Retrieval metrics accumulators
        total_hits_at_1 = 0
        total_hits_at_5 = 0
        total_hits_at_10 = 0
        total_retrieval_queries = 0

        for batch in iter_data:
            video_features = self._extract_video_tensor(batch)

            has_text_features = isinstance(batch, dict) and (
                'text_embs' in batch or 'text_emb' in batch
            )

            # Extract text embeddings if available
            if has_text_features:
                if 'text_embs' in batch:
                    text_embs = batch['text_embs'].to(self.device)
                else:
                    text_emb = batch['text_emb'].to(self.device)
                    text_embs = text_emb.unsqueeze(1)  # [batch_size, 1, text_dim]

            # VideoRQVAE_V2: returns 5 values (reconstructed, rq_loss, indices, x_encoded, x_decoded)
            recon_video_features, rq_loss, indices, encoder_out, x_decoded = self.model(
                video_features, text_embs if has_text_features else None, use_sk=False)

            # Reconstruction loss
            _, loss_recon = self.model.compute_loss(
                recon_video_features, rq_loss, video_features)
            recon_loss_list.append(loss_recon.item())

            # Contrastive loss and retrieval metrics if text available
            if has_text_features:
                loss_contrastive, hits_1, hits_5, hits_10, num_queries = \
                    self.model.compute_contrastive_loss_and_retrieval_test(x_decoded, text_embs)
                text_recon_loss_list.append(loss_contrastive.item())

                # Accumulate retrieval hits
                total_hits_at_1 += hits_1
                total_hits_at_5 += hits_5
                total_hits_at_10 += hits_10
                total_retrieval_queries += num_queries

            # Collision metrics (always computed)
            tokens_per_video, duplicates_per_video = self._accumulate_collision_counts(indices, token_counts)
            total_tokens += sum(tokens_per_video)
            inter_duplicate_counts.extend(duplicates_per_video)

        # Calculate average losses
        recon_loss = np.mean(recon_loss_list) if recon_loss_list else 0.0
        text_recon_loss = np.mean(text_recon_loss_list) if text_recon_loss_list else 0.0

        # Compute recall metrics
        recall_at_1 = total_hits_at_1 / total_retrieval_queries if total_retrieval_queries > 0 else 0.0
        recall_at_5 = total_hits_at_5 / total_retrieval_queries if total_retrieval_queries > 0 else 0.0
        recall_at_10 = total_hits_at_10 / total_retrieval_queries if total_retrieval_queries > 0 else 0.0

        metrics = self._finalize_collision_metrics(total_tokens, inter_duplicate_counts, token_counts)

        return metrics, recon_loss, text_recon_loss, recall_at_1, recall_at_5, recall_at_10

    def _save_checkpoint(self, epoch, collision_rate=1, test_recon_loss=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_recon_%.4f_model.pth' % (epoch, collision_rate, test_recon_loss))
        state = {
            "model_config": self.model.get_config(),
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_test_recon_loss": self.best_test_loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # Save scheduler state if scheduler is being used
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, and scheduler state from checkpoint."""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler state if it exists and scheduler is initialized
        if "scheduler" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.logger.info("Scheduler state loaded from checkpoint")

        # Load training metrics
        self.best_loss = checkpoint.get("best_loss", np.inf)
        self.best_test_loss = checkpoint.get("best_test_recon_loss", np.inf)
        self.best_recall_at_1 = checkpoint.get("best_recall_at_1", 0.0)
        self.best_recall_at_5 = checkpoint.get("best_recall_at_5", 0.0)
        self.best_recall_at_10 = checkpoint.get("best_recall_at_10", 0.0)

        epoch = checkpoint.get("epoch", 0)
        self.logger.info(f"Checkpoint loaded: epoch {epoch}, best_loss: {self.best_loss:.6f}")
        return epoch

    def fit(self, data_loader, valid_data_loader, test_data_loader):
        # Initialize learning rate scheduler
        self.scheduler = self._build_scheduler(data_loader)
        if self.scheduler:
            self.logger.info(f"Initialized {self.lr_scheduler_type} scheduler with warmup")

        # training loop
        global_step = 0
        log_interval = max(1, len(data_loader) // 5)

        for epoch in range(self.epochs):
            self.model.train()

            # Step-level tracking
            step_losses = []
            step_recon_losses = []
            step_text_losses = [] if self.text_guided else None
            step_quant_losses = []

            for batch_idx, batch_data in enumerate(data_loader):
                step_start_time = time()

                # Data processing based on mode
                if self.text_guided:
                    # Single-text guided mode
                    video_data = batch_data[self.video_key].detach().to(self.device)
                    if 'text_embs' in batch_data:
                        text_embs = batch_data['text_embs'].detach().to(self.device)
                    else:
                        text_embs = batch_data['text_emb'].detach().to(self.device)
                else:
                    # Video-only mode
                    video_data = batch_data.detach().to(self.device)
                    text_embs = None

                self.optimizer.zero_grad()

                # Forward pass based on mode
                if getattr(self, 'multi_text_mode', False):
                    # VideoRQVAE_V2 with contrastive learning
                    # forward() returns: (reconstructed, rq_loss, indices, x_encoded, x_decoded)
                    out, quant_loss, indices, encoder_out, x_decoded = self.model(
                        video_data, text_embs, use_sk=True)

                    # Compute reconstruction loss
                    loss_total, loss_recon = self.model.compute_loss(
                        out, quant_loss, video_data)

                    # Compute contrastive loss (requires semantic group ids)
                    text_group_ids = batch_data.get('text_group_ids')
                    if text_group_ids is None:
                        raise ValueError(
                            "Multi-text training with VideoRQVAE_V2 expects 'text_group_ids' in the batch."
                        )
                    text_group_ids = text_group_ids.detach().to(self.device)
                    
                    # Extract text_masks if available (for variable-length text batches)
                    text_masks = batch_data.get('text_masks', None)
                    if text_masks is not None:
                        text_masks = text_masks.detach().to(self.device)
                    
                    loss_contrastive = self.model.compute_contrastive_loss(
                        x_decoded, text_embs, text_group_ids, text_masks
                    )
                    loss_total = loss_total + self.contrastive_loss_weight * loss_contrastive

                    # Track losses
                    step_text_losses.append(loss_contrastive.item())

                else:
                    # Video-only mode
                    # VideoRQVAE_V2: (recon, quant_loss, indices, encoder_out, x_decoded)
                    recon_vemb, quant_loss, indices, encoder_out, x_decoded = self.model(video_data, use_sk=True)
                    # Compute reconstruction loss
                    loss_total, loss_recon = self.model.compute_loss(
                        recon_vemb, quant_loss, video_data)

                self._check_nan(loss_total)
                loss_total.backward()
                self.optimizer.step()

                # Step learning rate scheduler
                if self.scheduler:
                    self.scheduler.step()

                # Track losses for step-level logging
                step_losses.append(loss_total.item())
                step_recon_losses.append(loss_recon.item())
                step_quant_losses.append(quant_loss.item())

                global_step += 1
                step_end_time = time()

                # Log every log_interval steps
                if global_step % log_interval == 0:
                    # Calculate averages over the last log_interval steps
                    recent_steps = min(log_interval, len(step_losses))
                    avg_loss = np.mean(step_losses[-recent_steps:])
                    avg_recon_loss = np.mean(step_recon_losses[-recent_steps:])
                    avg_quant_loss = np.mean(step_quant_losses[-recent_steps:])
                    avg_text_loss = np.mean(step_text_losses[-recent_steps:]) if self.text_guided else None

                    # Build step-level log message dynamically
                    log_msg = f"Step: {global_step:6d} | Epoch: {epoch:3d} | Loss: {avg_loss:.6f} | Recon: {avg_recon_loss:.6f}"
                    if self.text_guided:
                        log_msg += f" | Text: {avg_text_loss:.6f}"
                    log_msg += f" | Quant: {avg_quant_loss:.6f} | Time: {step_end_time - step_start_time:.3f}s"
                    self.logger.info(log_msg)

                    # Step-level wandb logging
                    if self.use_wandb:
                        import wandb
                        log_dict = {
                            'train/total_loss': avg_loss,
                            'train/recon_loss': avg_recon_loss,
                            'train/quant_loss': avg_quant_loss
                        }
                        if self.text_guided:
                            log_dict['train/text_loss'] = avg_text_loss

                        # Add learning rate logging
                        current_lrs = self.scheduler.get_last_lr() if self.scheduler else [self.lr, self.vq_lr]
                        log_dict['lr/lr_main'] = current_lrs[0]
                        log_dict['lr/lr_vq'] = current_lrs[1] if len(current_lrs) > 1 else self.vq_lr

                        wandb.log(log_dict, step=global_step)

                    # Update best loss
                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss

            # Evaluation using updated collision metrics
            if epoch == 0 or epoch % self.eval_step == 0:
                valid_start_time = time()
                eval_metrics = self._valid_epoch(valid_data_loader)

                test_metrics, test_recon_loss, test_text_recon_loss, recall_1, recall_5, recall_10 = self._test_epoch(test_data_loader)
                test_total_loss = test_recon_loss + test_text_recon_loss

                # Log collision rate to wandb
                if self.use_wandb:
                    import wandb
                    log_dict = {
                        'eval/inter_collision_rate': eval_metrics.inter_collision_rate,
                        'eval/total_collision_rate': eval_metrics.total_collision_rate,
                        'eval/inter_duplicate_max': eval_metrics.inter_duplicate_max,
                        'eval/total_freq_max': eval_metrics.total_freq_max,
                        'test/inter_collision_rate': test_metrics.inter_collision_rate,
                        'test/total_collision_rate': test_metrics.total_collision_rate,
                        'test/inter_duplicate_max': test_metrics.inter_duplicate_max,
                        'test/total_freq_max': test_metrics.total_freq_max,
                        'test/total_loss': test_total_loss,
                        'test/recon_loss': test_recon_loss,
                        'test/text_recon_loss': test_text_recon_loss,
                        'test/recall@1': recall_1,
                        'test/recall@5': recall_5,
                        'test/recall@10': recall_10
                    }
                    wandb.log(log_dict)
                
                if test_total_loss < self.best_test_loss:
                    self.best_test_loss = test_total_loss
                    self._save_checkpoint(
                        epoch,
                        test_recon_loss=test_total_loss,
                        collision_rate=eval_metrics.total_collision_rate,
                        ckpt_file=self.best_test_recon_loss_ckpt,
                    )

                if recall_1 > self.best_recall_at_1:
                    self.best_recall_at_1 = recall_1
                    self._save_checkpoint(
                        epoch,
                        collision_rate=eval_metrics.total_collision_rate,
                        test_recon_loss=test_total_loss,
                        ckpt_file=self.best_recall_at_1_ckpt,
                    )

                if recall_5 > self.best_recall_at_5:
                    self.best_recall_at_5 = recall_5
                    self._save_checkpoint(
                        epoch,
                        collision_rate=eval_metrics.total_collision_rate,
                        test_recon_loss=test_total_loss,
                        ckpt_file=self.best_recall_at_5_ckpt,
                    )

                if recall_10 > self.best_recall_at_10:
                    self.best_recall_at_10 = recall_10
                    self._save_checkpoint(
                        epoch,
                        collision_rate=eval_metrics.total_collision_rate,
                        test_recon_loss=test_total_loss,
                        ckpt_file=self.best_recall_at_10_ckpt,
                    )

                # Validation results
                validation_msg = (
                    f"{set_color('Validation Results', 'yellow')} | "
                    f"InterDupMax: {eval_metrics.inter_duplicate_max} | "
                    f"InterCollision: {eval_metrics.inter_collision_rate:.6f} | "
                    f"TotalMax: {eval_metrics.total_freq_max} | "
                    f"TotalCollision: {eval_metrics.total_collision_rate:.6f}"
                )
                self.logger.info(validation_msg)

                # Test results
                test_msg = (
                    f"{set_color('Test Results', 'yellow')} | "
                    f"InterDupMax: {test_metrics.inter_duplicate_max} | "
                    f"InterCollision: {test_metrics.inter_collision_rate:.6f} | "
                    f"TotalMax: {test_metrics.total_freq_max} | "
                    f"TotalCollision: {test_metrics.total_collision_rate:.6f} | "
                    f"VideoReconLoss: {test_recon_loss:.6f} | "
                    f"TextReconLoss: {test_text_recon_loss:.6f} | "
                    f"R@1: {recall_1:.4f} | R@5: {recall_5:.4f} | R@10: {recall_10:.4f}"
                )
                self.logger.info(test_msg)
                
                valid_end_time = time()
                valid_score_output = (
                    f"{set_color('epoch %d evaluating' % epoch, 'green')} "
                    f"[{set_color('time', 'blue')}: {valid_end_time - valid_start_time:.2f}s, "
                    f"{set_color('current_collision', 'blue')}: {eval_metrics.total_collision_rate:.6f}, "
                    f"{set_color('current_total_loss', 'blue')}: {test_total_loss:.6f}]"
                )
                self.logger.info(valid_score_output)
        
        return self.best_loss, self.best_recall_at_10
    
