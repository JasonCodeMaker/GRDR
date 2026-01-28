from config.base_config import Config
import os
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id
from datasets.candidate_dataset import CandidateDataLoader
from tqdm import tqdm


def normalize_video_id_for_cache(video_id, dataset_name):
    """Convert video ID from loader format to cache filename format.

    ACTNET and DIDEMO video IDs include .mp4 suffix in the loader,
    but cache files are named without the suffix.
    """
    if dataset_name in ['ACTNET', 'DIDEMO']:
        if video_id.endswith('.mp4'):
            return video_id[:-4]
    return video_id


def load_cached_video_features(video_ids, cache_dir, dataset_name, is_clip4clip=False):
    """Load pre-cached video features from disk.

    Args:
        video_ids: List of video IDs (in loader format)
        cache_dir: Path to cache directory (e.g., reranker/xpool/video_features_cache)
        dataset_name: Dataset name for subdirectory selection
        is_clip4clip: If True, loads 'video_embed' (already pooled); else 'frame_embeds' (unpooled)

    Returns:
        Tuple of (features_tensor, video_ids_list, is_pooled)
        - features_tensor: [num_videos, embed_dim] (if pooled) or [num_videos, num_frames, embed_dim] (if unpooled)
        - video_ids_list: List of video IDs (in loader format, preserving original IDs)
        - is_pooled: Boolean indicating if features are already pooled

    Raises:
        FileNotFoundError: If cache directory or any cache file is missing
        ValueError: If embedding dimensions don't match expected shape
    """
    dataset_cache_dir = os.path.join(cache_dir, dataset_name)

    if not os.path.exists(dataset_cache_dir):
        raise FileNotFoundError(
            f"Cache directory not found: {dataset_cache_dir}\n"
            f"Please ensure video features are cached for dataset '{dataset_name}'"
        )

    features_list = []
    valid_video_ids = []

    for vid in tqdm(video_ids, desc=f"Loading cached features for {dataset_name}"):
        cache_vid = normalize_video_id_for_cache(vid, dataset_name)
        cache_file = os.path.join(dataset_cache_dir, f"{cache_vid}.npz")

        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Cache file not found for video '{vid}' (cache_vid='{cache_vid}')\n"
                f"Expected path: {cache_file}"
            )

        data = np.load(cache_file)
        
        if is_clip4clip:
            # CLIP4clip: features are already pooled, load 'video_embed'
            video_embed = data['video_embed']
            if video_embed.shape != (512,):
                raise ValueError(
                    f"Unexpected embedding shape for video '{vid}': {video_embed.shape}\n"
                    f"Expected: (512,)"
                )
            features_list.append(torch.from_numpy(video_embed))
        else:
            # Xpool: features are frame-level, load 'frame_embeds'
            frame_embeds = data['frame_embeds']
            if frame_embeds.shape != (12, 512):
                raise ValueError(
                    f"Unexpected embedding shape for video '{vid}': {frame_embeds.shape}\n"
                    f"Expected: (12, 512)"
                )
            features_list.append(torch.from_numpy(frame_embeds))
        
        valid_video_ids.append(vid)

    features_tensor = torch.stack(features_list)
    return features_tensor, valid_video_ids, is_clip4clip


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None,
                 expanded_pool_loader=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.expanded_pool_loader = expanded_pool_loader

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0

    def validate(self):
        """
        Validate the model.

        If expanded_pool_loader is set, includes train videos in search pool.
        """
        return self._valid_epoch_step(0, 0, 0, expanded_pool_loader=self.expanded_pool_loader)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
            
            data['video'] = data['video'].to(self.device)

            text_embeds, video_embeds_pooled = self.model(data)
            output = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            
            loss = self.loss(output, self.model.clip.logit_scale)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    self._save_checkpoint(epoch, save_best=True)

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps, pool_batch_size=64,
                          expanded_pool_loader=None):
        """
        Validate at a step when training an epoch at a certain step.

        Uses batched pool_frames computation to avoid O(N*M) memory explosion.

        Args:
            epoch: Current epoch number
            step: Current step within the epoch
            num_steps: Total number of steps in epoch
            pool_batch_size: Number of texts to process at once in pool_frames (default: 64)
                            Reduce this value if running out of memory
            expanded_pool_loader: Optional DataLoader with additional videos (e.g., training set)
                                 to include in the search pool for expanded evaluation

        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        vid_embed_pooled_arr = []  # Collect pre-computed pooled embeddings
        all_vid_ids = []

        with torch.no_grad():
            # Step 1: Collect all embeddings from test set
            for _, data in tqdm(enumerate(self.valid_data_loader), desc="Collecting test embeddings"):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)

                text_embed, vid_embed, vid_embed_pooled = self.model(data, return_all_frames=True)
                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                vid_embed_pooled_arr.append(vid_embed_pooled.cpu())  # Store pooled embeddings
                sims_batch = sim_matrix_training(text_embed, vid_embed_pooled, self.pooling_type)

                curr_loss = self.loss(sims_batch, self.model.clip.logit_scale)
                total_val_loss += curr_loss.item()

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)
            if self.pooling_type == 'avg':
                vid_embeds_pooled_cached = torch.cat(vid_embed_pooled_arr)  # Concatenate pre-computed pooled embeddings

            # Free intermediate lists
            del text_embed_arr, vid_embed_arr, vid_embed_pooled_arr

            # Store test video IDs for GT index mapping (before potentially adding train videos)
            test_vid_ids = all_vid_ids.copy()
            num_test_vids = len(test_vid_ids)

            # Step 1b: Collect train video embeddings if expanded pool is requested
            gt_indices = None
            if expanded_pool_loader is not None:
                # Determine cache directory based on architecture
                is_clip4clip = self.config.arch == "clip_baseline"
                if is_clip4clip:
                    cache_dir = 'reranker/xpool/video_features_cache/CLIP4clip'
                    print("Using CLIP4clip cache (already pooled features)")
                    # CLIP4clip only works with avg pooling since features are pre-pooled
                    if self.pooling_type != 'avg':
                        raise ValueError(
                            f"CLIP4clip (arch=clip_baseline) requires pooling_type='avg', "
                            f"but got pooling_type='{self.pooling_type}'. "
                            f"CLIP4clip features are pre-pooled and cannot be used with text-conditioned pooling."
                        )
                else:
                    cache_dir = 'reranker/xpool/video_features_cache/Xpool'
                    print("Using Xpool cache (frame-level features)")

                # Load cached video features instead of extracting on-the-fly
                train_vid_ids_raw = expanded_pool_loader.dataset.video_ids

                train_vid_embeds, train_vid_ids, is_pooled = load_cached_video_features(
                    train_vid_ids_raw,
                    cache_dir,
                    self.config.dataset_name,
                    is_clip4clip=is_clip4clip
                )

                # Handle pooling based on feature type
                if is_pooled:
                    # CLIP4clip: features are already pooled [num_videos, embed_dim]
                    # No need to pool, just concatenate with test video pooled embeddings
                    vid_embeds_pooled_cached = torch.cat([vid_embeds_pooled_cached, train_vid_embeds], dim=0)
                    # Note: vid_embeds is not updated for CLIP4clip since we only use pooled features
                    print(f"Loaded {len(train_vid_ids)} pre-pooled CLIP4clip train videos")
                else:
                    # Xpool: features are frame-level [num_videos, num_frames, embed_dim]
                    # Combine frame-level features
                    vid_embeds = torch.cat([vid_embeds, train_vid_embeds], dim=0)
                    
                    # For avg pooling, also pool the train video embeddings
                    if self.pooling_type == 'avg':
                        train_vid_embeds_pooled = train_vid_embeds.mean(dim=1)
                        vid_embeds_pooled_cached = torch.cat([vid_embeds_pooled_cached, train_vid_embeds_pooled], dim=0)
                        del train_vid_embeds_pooled
                    print(f"Loaded {len(train_vid_ids)} frame-level Xpool train videos")

                all_vid_ids = all_vid_ids + train_vid_ids

                # Build GT index mapping (test_vid_ids[i] -> position in combined pool)
                vid_to_idx = {v: i for i, v in enumerate(all_vid_ids)}
                gt_indices = torch.tensor([vid_to_idx[test_vid_ids[i]]
                                           for i in range(num_test_vids)])

                del train_vid_embeds
                print(f"Expanded pool: {num_test_vids} test + {len(train_vid_ids)} train = {len(all_vid_ids)} total videos")

            num_texts = text_embeds.shape[0]
            num_vids = vid_embeds.shape[0]

            # Get candidate mask if available (for candidate reranking mode)
            candidate_mask = getattr(self.valid_data_loader.dataset, 'candidate_mask', None)
            if candidate_mask is not None:
                candidate_mask = candidate_mask.to('cpu')

            if self.pooling_type == 'avg':
                # For avg pooling: use pre-computed pooled embeddings (no text conditioning)
                # vid_embeds_pooled_cached: [num_vids, embed_dim] (already computed by model)
                vid_embeds_pooled = vid_embeds_pooled_cached

                # Normalize embeddings
                text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                vid_embeds_pooled_norm = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

                # Simple cosine similarity: [num_texts, num_vids]
                sims = torch.mm(text_embeds_norm, vid_embeds_pooled_norm.t())

                # Apply candidate mask if available
                if candidate_mask is not None:
                    # Validate mask dimensions before applying
                    assert candidate_mask.shape[0] == sims.shape[0], \
                        f"Mask dim 0 ({candidate_mask.shape[0]}) != sims dim 0 ({sims.shape[0]})"
                    assert candidate_mask.shape[2] == sims.shape[1], \
                        f"Mask dim 2 ({candidate_mask.shape[2]}) != sims dim 1 ({sims.shape[1]})"
                    candidate_mask = candidate_mask.squeeze(1)
                    sims = sims.masked_fill(~candidate_mask, float('-inf'))

                del vid_embeds_pooled, vid_embeds, vid_embeds_pooled_norm, vid_embeds_pooled_cached
            else:
                # For text-conditioned pooling (topk/attention/transformer)
                # Memory-efficient batch-wise similarity computation
                # Instead of accumulating [V, T, D] tensor (~150GB for expanded pool),
                # compute similarities directly in batches to avoid OOM
                self.model.pool_frames.cpu()

                # Pre-allocate similarity matrix: [num_texts, num_vids]
                sims = torch.zeros(num_texts, num_vids, dtype=torch.float32)

                # Process text queries in batches
                for start_idx in tqdm(range(0, num_texts, pool_batch_size), desc="Pooling frames and computing similarities"):
                    end_idx = min(start_idx + pool_batch_size, num_texts)
                    text_batch = text_embeds[start_idx:end_idx]  # [batch_size, embed_dim]

                    # Normalize text batch
                    text_batch_norm = text_batch / text_batch.norm(dim=-1, keepdim=True)

                    # pool_frames returns [num_vids, batch_size, embed_dim]
                    pooled_batch = self.model.pool_frames(text_batch, vid_embeds)
                    
                    # Normalize pooled batch (each [v, b, :] vector independently)
                    pooled_batch_norm = pooled_batch / pooled_batch.norm(dim=-1, keepdim=True)

                    # Compute partial similarity: [batch_size, num_vids]
                    # einsum('bd,vbd->bv'): for each text b and video v, dot product over d
                    sims_batch = torch.einsum('bd,vbd->bv', text_batch_norm, pooled_batch_norm)
                    sims[start_idx:end_idx, :] = sims_batch

                    # Free intermediate tensors immediately
                    del pooled_batch, pooled_batch_norm, text_batch_norm, sims_batch

                self.model.pool_frames.cuda()

                # Apply candidate mask if available
                if candidate_mask is not None:
                    # Validate mask dimensions before applying
                    assert candidate_mask.shape[0] == sims.shape[0], \
                        f"Mask dim 0 ({candidate_mask.shape[0]}) != sims dim 0 ({sims.shape[0]})"
                    assert candidate_mask.shape[2] == sims.shape[1], \
                        f"Mask dim 2 ({candidate_mask.shape[2]}) != sims dim 1 ({sims.shape[1]})"
                    candidate_mask = candidate_mask.squeeze(1)
                    sims = sims.masked_fill(~candidate_mask, float('-inf'))

                del vid_embeds

            # Compute ranks
            sims_sort = torch.argsort(sims, dim=-1, descending=True)
            sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

            if gt_indices is not None:
                # Expanded pool: extract rank at GT position for each query
                ranks = sims_sort_2[torch.arange(len(gt_indices)), gt_indices].numpy()
            else:
                # Original: ground truth for text_i is video_i (diagonal)
                ranks = torch.diag(sims_sort_2).numpy()
            
            # Compute metrics
            from modules.metrics import compute_metrics
            res = compute_metrics(ranks)

            total_val_loss = total_val_loss / len(self.valid_data_loader)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res
