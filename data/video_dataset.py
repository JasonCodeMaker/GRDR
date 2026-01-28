import torch
from torch.utils.data import Dataset
import json
import os
import random
import numpy as np
from collections import defaultdict
from utils.data_utils import load_or_compute_kmeans_cache


def sample_validation_from_train(dataset_name, num_samples=1000, seed=42):
    """
    Sample unique video-caption pairs from training set for validation.
    Each unique video is sampled only once with a random caption.

    Args:
        dataset_name: Dataset name (msrvtt, activitynet, didemo, lsmdc)
        num_samples: Number of unique videos to sample (default 1000)
        seed: Random seed for reproducibility

    Returns:
        List of annotation dicts with 'video', 'caption' keys
    """
    dataset_name = dataset_name.lower()

    # Define annotation file paths for training data
    annotation_paths = {
        'msrvtt': './data/msrvtt/video_retreival_caption/msrvtt_ret_train.json',
        'actnet': './data/actnet/video_retreival_caption/actnet_ret_train.json',
        'didemo': './data/didemo/video_retreival_caption/didemo_ret_train.json',
        'lsmdc': './data/lsmdc/video_retreival_caption/lsmdc_ret_train.json',
    }

    if dataset_name not in annotation_paths:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         f"Supported: {list(annotation_paths.keys())}")

    annotation_file = annotation_paths[dataset_name]
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Training annotation file not found: {annotation_file}")

    # Load training annotations
    with open(annotation_file, 'r') as f:
        train_annotations = json.load(f)

    # Group annotations by video_id
    video_to_samples = defaultdict(list)
    for idx, item in enumerate(train_annotations):
        # Remove extension and extract filename (LSMDC has 'dir/filename' format)
        video_id = item['video'].replace('.mp4', '').replace('.avi', '')
        if '/' in video_id:
            video_id = video_id.split('/')[-1]
        video_to_samples[video_id].append((idx, item))

    # Sample unique videos
    random.seed(seed)
    unique_videos = list(video_to_samples.keys())
    selected_videos = random.sample(unique_videos, min(num_samples, len(unique_videos)))

    # Pick one random caption per video
    val_annotations = []
    for video_id in selected_videos:
        idx, item = random.choice(video_to_samples[video_id])
        val_annotations.append({
            'video': item['video'],
            'caption': item['caption'],
        })

    return val_annotations


class VideoTextDataset(Dataset):
    """
    Dataset for run.py that loads video features + text captions.
    Compatible with run.py's training pipeline (no dual tokenizer needed).

    This dataset:
    - Loads InternVideo2 video features from pickle files
    - Loads text features for optional usage
    - Returns raw captions for tokenization in collate_fn
    - One sample per caption (multiple captions per video)
    """

    def __init__(self, dataset_name, video_features, text_features,
                 tokenizer, split='train', max_text_len=128,
                 num_latent_tokens=4, cache_dir='./cache',
                 ids=None, aux_ids=None, use_pseudo_queries=False):
        """
        Args:
            dataset_name: 'msrvtt', 'actnet', etc.
            video_features: Dict[video_id -> np.array[512]]
            text_features: Dict[text_key -> np.array] (optional, for future use)
            tokenizer: Standard T5/BERT tokenizer (single tokenizer!)
            split: 'train', 'test', or 'val' (val samples 1000 unique videos from train)
            max_text_len: Max sequence length for captions
            num_latent_tokens: Number of latent tokens in VideoRQVAE (for k-means grouping)
            cache_dir: Directory for k-means cache
            ids: Dict[sample_key -> List[int]] hierarchical codes (e.g., {'video0_0': [0, 10, 5]})
            aux_ids: Auxiliary IDs (optional, for backward compatibility)
            use_pseudo_queries: If True, combine original and pseudo query files for training
        """
        self.dataset_name = dataset_name.lower()
        self.video_features = video_features
        self.text_features = text_features
        self.tokenizer = tokenizer
        self.split = split
        self.max_text_len = max_text_len
        self.num_latent_tokens = num_latent_tokens
        self.aux_ids = aux_ids
        self.use_pseudo_queries = use_pseudo_queries

        # Load caption annotations
        self.samples = self._load_annotations()

        print(f"[VideoTextDataset] Loaded {len(self.samples)} {split} samples for {dataset_name}")

        # Load or compute k-means text groupings (for token_idx assignment)
        # For validation split, use train k-means to ensure consistency
        import logging
        logger = logging.getLogger(__name__)
        if self.split != 'test':
            kmeans_split = 'train' if split == 'val' else split
            self.text_groups = load_or_compute_kmeans_cache(
                dataset_name, kmeans_split, video_features, text_features,
                num_latent_tokens, cache_dir, logger,
                use_pseudo_queries=self.use_pseudo_queries
            )
        else:
            self.text_groups = None

        # Store hierarchical codes (dict mapping video_id to code lists)
        if ids is None:
            self.video_codes = {s['video_id']: [0] for s in self.samples}
        else:
            self.video_codes = ids

    def _load_annotations(self):
        """Load caption annotations from JSON files"""

        # Handle validation split: sample from training data
        if self.split == 'val':
            annotations = sample_validation_from_train(
                self.dataset_name, num_samples=1000, seed=42
            )
            # Build samples from pre-sampled validation annotations
            samples = []
            missing_videos = set()
            video_caption_counts = {}  # Track caption count per raw video_id
            for item in annotations:
                # Remove extension and extract filename (LSMDC has 'dir/filename' format)
                raw_video_id = item['video'].replace('.mp4', '').replace('.avi', '')
                if '/' in raw_video_id:
                    raw_video_id = raw_video_id.split('/')[-1]
                caption = item['caption']

                if raw_video_id not in self.video_features:
                    missing_videos.add(raw_video_id)
                    continue

                # Handle list vs string caption format (Actnet/DiDeMo vs MSRVTT/LSMDC)
                # For validation, treat like training: pick one caption from list
                if isinstance(caption, list):
                    # Pick first caption from list for validation
                    cap = caption[0].strip() if caption else ""
                    count = video_caption_counts.get(raw_video_id, 0)
                    video_id = f"{raw_video_id}_{count}"
                    video_caption_counts[raw_video_id] = count + 1
                    samples.append({
                        'video_id': video_id,
                        'caption': cap,
                    })
                else:
                    # Single caption format (MSRVTT, LSMDC)
                    count = video_caption_counts.get(raw_video_id, 0)
                    video_id = f"{raw_video_id}_{count}"
                    video_caption_counts[raw_video_id] = count + 1
                    samples.append({
                        'video_id': video_id,
                        'caption': caption,
                    })

            if missing_videos:
                print(f"  Warning: {len(missing_videos)} videos not found in features, skipped")
            return samples

        # Define annotation file paths for different datasets
        annotation_paths = {
            'msrvtt': f'./data/msrvtt/video_retreival_caption/msrvtt_ret_{self.split}.json',
            'actnet': f'./data/actnet/video_retreival_caption/actnet_ret_{self.split}.json',
            'didemo': f'./data/didemo/video_retreival_caption/didemo_ret_{self.split}.json',
            'lsmdc': f'./data/lsmdc/video_retreival_caption/lsmdc_ret_{self.split}.json',
        }

        if self.dataset_name not in annotation_paths:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. "
                           f"Supported: {list(annotation_paths.keys())}")

        annotation_file = annotation_paths[self.dataset_name]

        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        original_count = len(annotations)

        # Combine with pseudo queries for training split (if enabled)
        if self.use_pseudo_queries and self.split == 'train':
            pseudo_paths = {
                'msrvtt': './data/msrvtt/video_retreival_caption/msrvtt_ret_train_pesudo.json',
                'actnet': './data/actnet/video_retreival_caption/actnet_ret_train_pesudo.json',
                'didemo': './data/didemo/video_retreival_caption/didemo_ret_train_pesudo.json',
                'lsmdc': './data/lsmdc/video_retreival_caption/lsmdc_ret_train_pesudo.json',
            }
            pseudo_path = pseudo_paths.get(self.dataset_name)
            if pseudo_path and os.path.exists(pseudo_path):
                with open(pseudo_path, 'r') as f:
                    pseudo_annotations = json.load(f)
                annotations.extend(pseudo_annotations)
                print(f"  Combined {original_count} original + {len(pseudo_annotations)} pseudo = {len(annotations)} total annotations")
            else:
                print(f"  Warning: Pseudo query file not found for {self.dataset_name}, using original only")

        # Build samples: one per caption with unique video_id (video7015_0 format)
        samples = []
        missing_videos = set()
        video_caption_counts = {}  # Track caption count per raw video_id

        for item in annotations:
            # Extract raw video ID (remove extension and directory path for LSMDC)
            raw_video_id = item['video'].replace('.mp4', '').replace('.avi', '')
            if '/' in raw_video_id:
                raw_video_id = raw_video_id.split('/')[-1]
            caption = item['caption']

            # Skip if video features don't exist
            if raw_video_id not in self.video_features:
                missing_videos.add(raw_video_id)
                continue

            # Handle list vs string caption format (Actnet/DiDeMo vs MSRVTT/LSMDC)
            if isinstance(caption, list):
                if self.split == 'train':
                    # Training: split list into separate samples, one per caption
                    for cap in caption:
                        count = video_caption_counts.get(raw_video_id, 0)
                        video_id = f"{raw_video_id}_{count}"
                        video_caption_counts[raw_video_id] = count + 1
                        samples.append({
                            'video_id': video_id,
                            'caption': cap.strip(),
                        })
                else:
                    # Test: concatenate all captions into one for 1:1 mapping
                    video_id = f"{raw_video_id}_0"
                    combined_caption = ' '.join(c.strip() for c in caption)
                    samples.append({
                        'video_id': video_id,
                        'caption': combined_caption,
                    })
            else:
                # Single caption format (MSRVTT, LSMDC)
                count = video_caption_counts.get(raw_video_id, 0)
                video_id = f"{raw_video_id}_{count}"
                video_caption_counts[raw_video_id] = count + 1
                samples.append({
                    'video_id': video_id,
                    'caption': caption,
                })

        if missing_videos:
            print(f"  Warning: {len(missing_videos)} videos not found in features, skipped")
            if len(missing_videos) <= 5:
                print(f"  Missing video IDs: {list(missing_videos)[:5]}")

        return samples

    def __len__(self):
        return len(self.samples)

    def getitem(self, idx):
        """
        Helper method to fetch a single sample.

        Returns dict with all sample information including hierarchical codes.
        """
        sample = self.samples[idx]
        video_id = sample['video_id']  # Unique ID (e.g., video7015_0)
        caption = sample['caption']

        # Extract raw video_id for video_features lookup
        raw_video_id = video_id.rsplit('_', 1)[0]  # video7015_0 -> video7015

        # Load video features (InternVideo2 pooled embeddings)
        video_emb = self.video_features[raw_video_id]

        # Convert to tensor if numpy array
        if isinstance(video_emb, np.ndarray):
            video_features = torch.from_numpy(video_emb).float()
        else:
            video_features = torch.tensor(video_emb).float()

        # Get k-means token assignment (used by trainer for label generation)
        # For test set, always use token 0; otherwise use k-means assignment
        if self.split == 'test':
            token_idx = 0
        else:
            token_idx = self.text_groups.get(video_id, 0)  # Default to token 0

        # Get hierarchical code for this sample (direct lookup by video_id)
        hierarchical_code = self.video_codes.get(video_id, [0])

        # Get auxiliary ID if available
        if self.aux_ids is None:
            aux_id = -100
        else:
            aux_id = self.aux_ids.get(video_id, -100)

        return {
            'caption_text': caption,
            'video_features': video_features,
            'video_id': video_id,
            'token_idx': token_idx,
            'hierarchical_code': hierarchical_code,
            'aux_id': aux_id,
        }

    def __getitem__(self, idx):
        """
        Returns sample ready for run.py's training pipeline.

        Output format:
        {
            'caption_text': str - raw caption text for tokenization
            'video_features': torch.Tensor [512] - InternVideo2 pooled features
            'video_id': str - unique video identifier (e.g., video7015_0)
            'token_idx': int - k-means assigned latent token index (0 to num_latent_tokens-1)
            'hierarchical_code': List[int] - hierarchical semantic code (e.g., [0, 10, 5])
            'aux_id': int - auxiliary ID (default -100)
        }
        """
        return self.getitem(idx)


def collate_fn(batch, tokenizer, max_length=128):
    """
    Collate function compatible with run.py batch adapter.

    This function:
    - Tokenizes caption texts using single T5 tokenizer
    - Stacks video features
    - Converts hierarchical codes to tensor
    """
    # Extract fields from batch samples
    caption_texts = [item['caption_text'] for item in batch]
    video_features = torch.stack([item['video_features'] for item in batch])
    video_ids = [item['video_id'] for item in batch]
    token_idx = [item['token_idx'] for item in batch]
    hierarchical_codes = [item['hierarchical_code'] for item in batch]
    aux_ids = [item['aux_id'] for item in batch]

    # Tokenize captions with single T5 tokenizer
    caption_encodings = tokenizer(
        caption_texts,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )

    # Convert hierarchical codes to tensor (pad to max code length in batch)
    # All codes in a batch should have the same length, but handle variable lengths
    if hierarchical_codes:
        max_code_len = max(len(code) for code in hierarchical_codes)
        padded_codes = []
        for code in hierarchical_codes:
            padded = code + [0] * (max_code_len - len(code))
            padded_codes.append(padded)
        ids_tensor = torch.tensor(padded_codes, dtype=torch.long)
    else:
        ids_tensor = torch.tensor([[0]], dtype=torch.long)

    # Convert aux_ids to tensor (or None if all are -100)
    if all(aux_id == -100 for aux_id in aux_ids):
        aux_ids_tensor = None
    else:
        aux_ids_tensor = torch.tensor(aux_ids, dtype=torch.long)

    return {
        'caption_tokens': caption_encodings['input_ids'],      # [B, seq_len]
        'attention_mask': caption_encodings['attention_mask'], # [B, seq_len]
        'video_features': video_features,                      # [B, 512]
        'video_ids': video_ids,                                # List[str] unique IDs
        'token_idx': torch.tensor(token_idx, dtype=torch.long),  # [B]
        'ids': ids_tensor,                                     # [B, code_length]
        'aux_ids': aux_ids_tensor,                             # [B] or None
    }
