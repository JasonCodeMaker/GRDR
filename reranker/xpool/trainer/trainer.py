from config.base_config import Config
import os
import json
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


def resolve_dataset_cache_dir(cache_dir, dataset_name):
    """Accept either a cache root or a dataset-specific cache directory."""
    normalized_cache_dir = os.path.normpath(cache_dir)
    if os.path.basename(normalized_cache_dir).upper() == dataset_name.upper():
        return normalized_cache_dir
    return os.path.join(normalized_cache_dir, dataset_name)


def load_cached_video_features(video_ids, cache_dir, dataset_name, is_clip4clip=False,
                               expected_num_frames=12):
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
    dataset_cache_dir = resolve_dataset_cache_dir(cache_dir, dataset_name)

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
            if frame_embeds.shape != (expected_num_frames, 512):
                raise ValueError(
                    f"Unexpected embedding shape for video '{vid}': {frame_embeds.shape}\n"
                    f"Expected: ({expected_num_frames}, 512)"
                )
            features_list.append(torch.from_numpy(frame_embeds))
        
        valid_video_ids.append(vid)

    features_tensor = torch.stack(features_list)
    return features_tensor, valid_video_ids, is_clip4clip


def infer_cached_num_frames(video_ids, cache_dir, dataset_name):
    """Inspect cached XPool frame embeddings and return cached frames/video."""
    dataset_cache_dir = resolve_dataset_cache_dir(cache_dir, dataset_name)

    if not os.path.exists(dataset_cache_dir):
        raise FileNotFoundError(
            f"Cache directory not found: {dataset_cache_dir}\n"
            f"Please ensure video features are cached for dataset '{dataset_name}'"
        )

    for vid in video_ids:
        cache_vid = normalize_video_id_for_cache(vid, dataset_name)
        cache_file = os.path.join(dataset_cache_dir, f"{cache_vid}.npz")
        if not os.path.exists(cache_file):
            continue

        data = np.load(cache_file)
        frame_embeds = data['frame_embeds']
        if frame_embeds.ndim != 2 or frame_embeds.shape[1] != 512:
            raise ValueError(
                f"Unexpected cached frame embedding shape for video '{vid}': {frame_embeds.shape}\n"
                f"Expected: (num_frames, 512)"
            )
        return frame_embeds.shape[0]

    raise FileNotFoundError(
        f"No cache files found under: {dataset_cache_dir}\n"
        f"Unable to infer cached frame count for dataset '{dataset_name}'"
    )


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
        self.no_improve_count = 0
        self.early_stop_patience = getattr(config, 'early_stop_patience', 0)

    def validate(self):
        """
        Validate the model.

        If expanded_pool_loader is set, includes train videos in search pool.
        """
        return self._valid_epoch_step(0, 0, 0, expanded_pool_loader=self.expanded_pool_loader)

    def _resolve_video_cache_root(self):
        if self.config.video_cache_dir:
            cache_root = self.config.video_cache_dir
        elif self.config.arch == "clip_baseline":
            cache_root = 'reranker/xpool/video_features_cache/CLIP4clip'
        else:
            cache_root = 'reranker/xpool/video_features_cache/Xpool'

        return cache_root, self.config.arch == "clip_baseline"

    def _validate_cache_frames(self, video_ids, cache_root, is_clip4clip, require_cache):
        if is_clip4clip:
            if self.pooling_type != 'avg':
                raise ValueError(
                    "CLIP4clip cached features are pre-pooled and only support avg pooling"
                )
            return True

        cached_num_frames = infer_cached_num_frames(video_ids, cache_root, self.config.dataset_name)
        if cached_num_frames == self.config.num_frames:
            return True

        if require_cache:
            raise ValueError(
                f"Cached features use {cached_num_frames} frames/video, "
                f"but config.num_frames={self.config.num_frames}"
            )
        return False

    def _load_cached_features_in_order(self, video_ids, cache_root, is_clip4clip):
        unique_video_ids = list(dict.fromkeys(video_ids))
        unique_features, _, is_pooled = load_cached_video_features(
            unique_video_ids,
            cache_root,
            self.config.dataset_name,
            is_clip4clip=is_clip4clip,
            expected_num_frames=self.config.num_frames,
        )

        if len(unique_video_ids) == len(video_ids):
            return unique_features.float(), video_ids, is_pooled

        feature_by_video_id = {
            vid: unique_features[idx] for idx, vid in enumerate(unique_video_ids)
        }
        ordered_features = torch.stack([feature_by_video_id[vid] for vid in video_ids]).float()
        return ordered_features, video_ids, is_pooled

    def _collect_eval_text_video_pairs(self):
        dataset = self.valid_data_loader.dataset

        if self.config.dataset_name == 'LSMDC':
            video_ids = list(dataset.clip2caption.keys())
            texts = [dataset.clip2caption[vid] for vid in video_ids]
            return texts, video_ids

        if self.config.dataset_name == 'MSRVTT':
            return dataset.test_df['sentence'].tolist(), dataset.test_df['video_id'].tolist()

        if self.config.dataset_name == 'MSVD':
            texts = [caption for _, caption in dataset.all_test_pairs]
            video_ids = [vid for vid, _ in dataset.all_test_pairs]
            return texts, video_ids

        if self.config.dataset_name in {'ACTNET', 'DIDEMO'}:
            texts = [caption for _, caption in dataset.all_pairs]
            video_ids = [
                normalize_video_id_for_cache(video_id, self.config.dataset_name)
                for video_id, _ in dataset.all_pairs
            ]
            return texts, video_ids

        if self.config.dataset_name == 'PANDA':
            from datasets.media_utils import strip_media_extension
            texts = [row['caption'] for row in dataset.entries]
            video_ids = [strip_media_extension(row['video']) for row in dataset.entries]
            return texts, video_ids

        raise NotImplementedError(
            f"Cached evaluation metadata extraction is not implemented for {self.config.dataset_name}"
        )

    def _encode_text_batch(self, batch_texts):
        if self.tokenizer is not None:
            text_batch = self.tokenizer(
                batch_texts, return_tensors='pt', padding=True, truncation=True
            )
        else:
            text_batch = batch_texts

        if isinstance(text_batch, torch.Tensor):
            text_batch = text_batch.to(self.device)
        else:
            text_batch = {key: val.to(self.device) for key, val in text_batch.items()}

        if self.config.huggingface:
            text_features = self.model.clip.get_text_features(**text_batch)
        else:
            text_features = self.model.clip.encode_text(text_batch)

        if self.config.arch == "clip_baseline":
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _collect_cached_test_embeddings(self, cache_root, is_clip4clip):
        test_texts, test_vid_ids = self._collect_eval_text_video_pairs()
        test_vid_embeds, all_vid_ids, is_pooled = self._load_cached_features_in_order(
            test_vid_ids, cache_root, is_clip4clip
        )

        text_embed_arr = []
        total_val_loss = 0.0
        num_batches = 0

        if is_pooled:
            vid_embeds = None
            vid_embeds_pooled_cached = test_vid_embeds
        else:
            vid_embeds = test_vid_embeds
            vid_embeds_pooled_cached = None
            if self.pooling_type == 'avg':
                vid_embeds_pooled_cached = vid_embeds.mean(dim=1)

        with torch.no_grad():
            for start_idx in tqdm(
                range(0, len(test_texts), self.config.batch_size),
                desc="Collecting cached test embeddings",
            ):
                end_idx = min(start_idx + self.config.batch_size, len(test_texts))
                text_batch = self._encode_text_batch(test_texts[start_idx:end_idx])
                text_embed_arr.append(text_batch.cpu())

                if is_pooled:
                    vid_embed_pooled_batch = vid_embeds_pooled_cached[start_idx:end_idx]
                elif self.pooling_type == 'avg':
                    vid_embed_pooled_batch = vid_embeds_pooled_cached[start_idx:end_idx]
                else:
                    vid_batch = vid_embeds[start_idx:end_idx].to(self.device)
                    vid_embed_pooled_batch = self.model.pool_frames(text_batch, vid_batch).cpu()
                    del vid_batch

                sims_batch = sim_matrix_training(
                    text_batch,
                    vid_embed_pooled_batch.to(self.device),
                    self.pooling_type,
                )
                total_val_loss += self.loss(sims_batch, self.model.clip.logit_scale).item()
                num_batches += 1

        text_embeds = torch.cat(text_embed_arr)
        return text_embeds, vid_embeds, vid_embeds_pooled_cached, all_vid_ids, total_val_loss, num_batches

    def _encode_video_batch(self, batch_video):
        batch_size = batch_video.shape[0]
        video_data = batch_video.reshape(-1, 3, self.config.input_res, self.config.input_res)

        if hasattr(self.model, 'forward_video'):
            video_features = self.model.forward_video(video_data)
        elif self.config.huggingface:
            video_features = self.model.clip.get_image_features(video_data)
        else:
            video_features = self.model.clip.encode_image(video_data)

        if self.config.arch == "clip_baseline":
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_features.reshape(batch_size, self.config.num_frames, -1)


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

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']
                    self._save_checkpoint(epoch, save_best=True)
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

                if self.early_stop_patience > 0 and self.no_improve_count >= self.early_stop_patience:
                    print(f"Early-stop: R@1 has not improved for {self.no_improve_count} consecutive evals (patience={self.early_stop_patience}); requesting stop after this epoch.")
                    self.stop_flag = True

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps, pool_batch_size=None,
                          expanded_pool_loader=None):
        if pool_batch_size is None:
            pool_batch_size = getattr(self.config, 'pool_batch_size', 64)
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
        all_vid_ids = []
        cache_eval_batches = None

        with torch.no_grad():
            use_cached_test_pool = getattr(self.config, 'use_cached_video_features', False)
            cache_root, is_clip4clip = self._resolve_video_cache_root()

            if use_cached_test_pool:
                test_texts, test_vid_ids = self._collect_eval_text_video_pairs()
                self._validate_cache_frames(
                    test_vid_ids,
                    cache_root,
                    is_clip4clip,
                    require_cache=True,
                )
                print("Using cached video features for test evaluation pool")
                text_embeds, vid_embeds, vid_embeds_pooled_cached, all_vid_ids, total_val_loss, cache_eval_batches = (
                    self._collect_cached_test_embeddings(cache_root, is_clip4clip)
                )
            else:
                text_embed_arr = []
                vid_embed_arr = []
                vid_embed_pooled_arr = []  # Collect pre-computed pooled embeddings

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
                    vid_embeds_pooled_cached = torch.cat(vid_embed_pooled_arr)

                del text_embed_arr, vid_embed_arr, vid_embed_pooled_arr

            # Store test video IDs for GT index mapping (before potentially adding train videos)
            test_vid_ids = all_vid_ids.copy()
            num_test_vids = len(test_vid_ids)

            # Step 1b: Collect train video embeddings if expanded pool is requested
            gt_indices = None
            if expanded_pool_loader is not None:
                use_cached_train_pool = True
                if is_clip4clip:
                    print("Using CLIP4clip cache (already pooled features)")
                else:
                    use_cached_train_pool = self._validate_cache_frames(
                        expanded_pool_loader.dataset.video_ids,
                        cache_root,
                        is_clip4clip,
                        require_cache=use_cached_test_pool,
                    )
                    if use_cached_train_pool:
                        print("Using Xpool cache (frame-level features)")
                    else:
                        print(
                            "Skipping Xpool cache for expanded pool because cached frame count "
                            f"does not match config.num_frames={self.config.num_frames}"
                        )

                train_vid_ids_raw = expanded_pool_loader.dataset.video_ids
                if use_cached_train_pool:
                    cache_root_resolved = resolve_dataset_cache_dir(cache_root, self.config.dataset_name)
                    filtered_train_ids = [
                        vid for vid in train_vid_ids_raw
                        if os.path.exists(os.path.join(
                            cache_root_resolved,
                            f"{normalize_video_id_for_cache(vid, self.config.dataset_name)}.npz"))
                    ]
                    n_missing = len(train_vid_ids_raw) - len(filtered_train_ids)
                    if n_missing:
                        print(f"Skipping {n_missing} train video(s) with missing cache entries "
                              f"({len(filtered_train_ids)}/{len(train_vid_ids_raw)} remain)")
                    train_vid_embeds, train_vid_ids, is_pooled = load_cached_video_features(
                        filtered_train_ids,
                        cache_root,
                        self.config.dataset_name,
                        is_clip4clip=is_clip4clip,
                        expected_num_frames=self.config.num_frames,
                    )
                    train_vid_embeds = train_vid_embeds.float()
                else:
                    train_vid_embed_arr = []
                    train_vid_ids = []
                    for _, train_data in tqdm(
                        enumerate(expanded_pool_loader),
                        desc="Collecting expanded-pool train embeddings"
                    ):
                        train_data['video'] = train_data['video'].to(self.device)
                        train_vid_embed = self._encode_video_batch(train_data['video'])
                        train_vid_embed_arr.append(train_vid_embed.cpu())
                        for v_id in train_data['video_id']:
                            train_vid_ids.append(v_id)

                    train_vid_embeds = torch.cat(train_vid_embed_arr)
                    del train_vid_embed_arr
                    is_pooled = False

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
            if self.pooling_type == 'avg':
                num_vids = vid_embeds_pooled_cached.shape[0]
            else:
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

            # Compute ranks (memory-efficient: skip sims_sort_2 when per-query export not requested,
            # since 5694 x 1.2M x int64 = ~53 GB peak — OOMs Nectar 117 GB host).
            per_query_path = getattr(self.config, "save_per_query_ranks", None)
            if per_query_path:
                sims_sort = torch.argsort(sims, dim=-1, descending=True)
                sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)
                if gt_indices is not None:
                    ranks = sims_sort_2[torch.arange(len(gt_indices)), gt_indices].numpy()
                else:
                    ranks = torch.diag(sims_sort_2).numpy()
            else:
                if gt_indices is not None:
                    gt_sims_vec = sims[torch.arange(len(gt_indices)), gt_indices]
                else:
                    gt_sims_vec = sims.diagonal()
                ranks = np.zeros(sims.shape[0], dtype=np.int64)
                for q in range(sims.shape[0]):
                    ranks[q] = int((sims[q] > gt_sims_vec[q]).sum())

            if per_query_path:
                per_query_results = []
                candidate_counts = None
                if candidate_mask is not None:
                    candidate_counts = candidate_mask.sum(dim=1).cpu().numpy().tolist()
                top_k = min(10, sims_sort.shape[1])
                top_indices = sims_sort[:, :top_k].cpu().numpy()
                for query_idx, rank in enumerate(ranks):
                    gt_video_id = test_vid_ids[query_idx] if query_idx < len(test_vid_ids) else ""
                    top_videos = []
                    for vid_idx in top_indices[query_idx]:
                        vid_id = all_vid_ids[int(vid_idx)]
                        score = float(sims[query_idx, int(vid_idx)].item())
                        if np.isfinite(score):
                            top_videos.append([vid_id, score])
                    per_query_results.append({
                        "query_idx": query_idx,
                        "candidate_query_idx": query_idx,
                        "video_id_gt": gt_video_id,
                        "rank": int(rank),
                        "candidate_count": int(candidate_counts[query_idx]) if candidate_counts is not None else len(all_vid_ids),
                        "top_videos": top_videos,
                    })
                os.makedirs(os.path.dirname(per_query_path), exist_ok=True)
                with open(per_query_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "per_query_results": per_query_results,
                        "config": {
                            "candidate_file": getattr(self.config, "candidate_file", None),
                            "index_safe_candidate_mask": getattr(self.config, "index_safe_candidate_mask", False),
                            "expanded_pool": getattr(self.config, "expanded_pool", False),
                        },
                    }, f, indent=2)
                print(f"Saved per-query ranks to: {per_query_path}")
            
            # Compute metrics
            from modules.metrics import compute_metrics
            res = compute_metrics(ranks)

            if cache_eval_batches is not None:
                total_val_loss = total_val_loss / max(1, cache_eval_batches)
            else:
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
