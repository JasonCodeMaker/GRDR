from torch.utils.data import Dataset
import random
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import PreTrainedTokenizerBase
import numpy as np

from .data_utils import (
    load_or_compute_kmeans_cache,
    load_caption_annotations,
    indices_to_string,
)


def extract_video_id(video_name, dataset_name):
    """Extract video ID from video filename, handling dataset-specific formats"""
    if dataset_name == 'lsmdc':
        # e.g., "3001_21_JUMP_STREET/3001_21_JUMP_STREET_00.02.55.644-00.02.56.718.avi"
        basename = video_name.split('/')[-1]
        return basename.replace('.avi', '')
    else:
        # msrvtt, actnet, didemo use .mp4
        return video_name.replace('.mp4', '')


def get_caption_text(item, dataset_name, is_train=True):
    """Extract caption text from item, handling dataset-specific formats
    
    For ACTNET and DIDEMO:
    - Training: Returns list of captions (each caption becomes separate training sample)
    - Test: Returns joined caption string (one query per video)
    
    For MSRVTT and LSMDC:
    - Always returns single caption string
    """
    caption = item['caption']
    
    if dataset_name in ["actnet", "didemo"]:
        # Both datasets have list of captions
        if isinstance(caption, list):
            if is_train:
                # Training: return list for expansion
                return caption
            else:
                # Test: join all captions with space, stripping each caption first to match reranker format
                return " ".join([cap.strip() if isinstance(cap, str) else str(cap) for cap in caption])
        return caption
    elif dataset_name in ["msrvtt", "lsmdc"]:
        # Single string caption
        return caption
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

class T5Dataset(Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        data_file=None,
        caption_file=None,
        index_file=None,
        max_source_len=128,
        max_target_len=32,
        add_prefix=False,
        subset_size=None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.subset_size = subset_size
        self.add_prefix = add_prefix

        self.source_texts = []
        self.target_texts = []

        # Determine format: combined (videorqvae) or separate (standard/text_guided)
        if data_file is not None:
            # Combined format: list with caption + SemanticID
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                caption = get_caption_text(item, self.dataset_name, is_train=True)
                
                # Convert SemanticID list to string (remove < > tokens)
                semantic_ids = item['SemanticID']
                target_text = ' '.join(semantic_ids).replace("<", "").replace(">", "")

                # For ACTNET/DIDEMO, caption is a list -> create separate sample for each
                if isinstance(caption, list):
                    for cap in caption:
                        self.source_texts.append(cap)
                        self.target_texts.append(target_text)
                else:
                    self.source_texts.append(caption)
                    self.target_texts.append(target_text)
        
        elif caption_file is not None and index_file is not None:
            # Separate format: caption file + index file
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            for item in caption_data:
                video_name = item['video']
                video_id = extract_video_id(video_name, self.dataset_name)
                
                # Only include videos that have semantic IDs
                if video_id in index_data:
                    caption = get_caption_text(item, self.dataset_name, is_train=True)
                    
                    semantic_ids = index_data[video_id]
                    target_text = ' '.join(semantic_ids).replace("<", "").replace(">", "")
                    
                    # For ACTNET/DIDEMO, caption is a list -> create separate sample for each
                    if isinstance(caption, list):
                        for cap in caption:
                            self.source_texts.append(cap)
                            self.target_texts.append(target_text)
                    else:
                        self.source_texts.append(caption)
                        self.target_texts.append(target_text)
        else:
            raise ValueError("Must provide either data_file or both caption_file and index_file")

        # Apply subset sampling if specified
        if self.subset_size is not None and self.subset_size < len(self.source_texts):
            indices = list(range(len(self.source_texts)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.source_texts = [self.source_texts[i] for i in sampled_indices]
            self.target_texts = [self.target_texts[i] for i in sampled_indices]

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        if self.add_prefix:
            source_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction:{source_text} Response:"
        target_text = self.target_texts[idx]

        # Encode source text with tokenizer
        source_encodings = self.tokenizer(
            source_text,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        )

        # Encode target with tokenizer
        # target_text is space-separated sID tokens like "a_0 b_1 c_2"
        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False  # Don't add EOS yet
        )

        # Truncate if needed (reserve space for EOS)
        if len(target_ids) > self.max_target_len - 1:
            target_ids = target_ids[:self.max_target_len - 1]

        # Add EOS token
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        # Pad to max_length
        pad_length = self.max_target_len - len(target_ids)
        target_ids = target_ids + [self.tokenizer.pad_token_id] * pad_length

        # Convert to tensor
        labels = torch.tensor(target_ids)
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encodings["input_ids"].squeeze(),
            "attention_mask": source_encodings["attention_mask"].squeeze(),
            "labels": labels
        }


class T5TestDataset(Dataset):
    """Test dataset for retrieval evaluation with multiple SemanticIDs per video"""

    def __init__(
        self,
        dataset_name,
        tokenizer,
        caption_file,
        index_file,
        max_source_len=128,
        max_target_len=32,
        add_prefix=False,
        return_all_targets=False,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.add_prefix = add_prefix
        self.return_all_targets = return_all_targets

        # Load caption data (list format: one caption per video)
        with open(caption_file, 'r', encoding='utf-8') as f:
            caption_data = json.load(f)

        # Load semantic ID index (dict format: video_id -> [SemanticID lists])
        with open(index_file, 'r', encoding='utf-8') as f:
            self.index_data = json.load(f)

        # Build test samples
        self.samples = []
        for item in caption_data:
            video_name = item['video']
            video_id = extract_video_id(video_name, self.dataset_name)

            # Only include videos that have semantic IDs
            if video_id in self.index_data:
                # For test, join captions for ACTNET/DIDEMO (one query per video)
                caption = get_caption_text(item, self.dataset_name, is_train=False)

                self.samples.append({
                    'caption': caption,
                    'video_id': video_id,
                    'video_name': video_name
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        source_text = sample['caption']

        if self.add_prefix:
            source_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction:{source_text} Response:"

        # Encode source with tokenizer
        source_encodings = self.tokenizer(
            source_text,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        )

        # Get all semantic ID lists for this video
        semantic_id_data = self.index_data[sample['video_id']]
        
        # Handle both formats: single list or list of lists
        if semantic_id_data and isinstance(semantic_id_data[0], list):
            # VideoRQVAE format: list of lists
            semantic_id_lists = semantic_id_data
        else:
            # Standard/text_guided format: single list, wrap it
            semantic_id_lists = [semantic_id_data]

        # Use first target as primary label (for compatibility)
        first_target = ' '.join(semantic_id_lists[0]).replace("<", "").replace(">", "")

        # Encode target with tokenizer
        target_ids = self.tokenizer.encode(
            first_target,
            add_special_tokens=False
        )

        # Truncate if needed (reserve space for EOS)
        if len(target_ids) > self.max_target_len - 1:
            target_ids = target_ids[:self.max_target_len - 1]

        # Add EOS token
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        # Pad to max_length
        pad_length = self.max_target_len - len(target_ids)
        target_ids = target_ids + [self.tokenizer.pad_token_id] * pad_length

        # Convert to tensor
        labels = torch.tensor(target_ids)
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": source_encodings["input_ids"].squeeze(),
            "attention_mask": source_encodings["attention_mask"].squeeze(),
            "labels": labels,
            "video_id": sample['video_id'],
        }

        # Convert all SemanticID lists to string format for multi-target evaluation
        target_texts = [
            ' '.join(sid_list).replace("<", "").replace(">", "")
            for sid_list in semantic_id_lists
        ]
        result["target_texts"] = target_texts

        return result


class CombinedT5Dataset(Dataset):
    """T5Dataset that combines multiple caption/index file pairs for expanded training data"""

    def __init__(
        self,
        dataset_name,
        tokenizer,
        caption_files,
        index_files,
        max_source_len=128,
        max_target_len=32,
        add_prefix=False,
        subset_size=None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.subset_size = subset_size
        self.add_prefix = add_prefix
        
        # Combine data from multiple file pairs
        self.source_texts = []
        self.target_texts = []
        
        for caption_file, index_file in zip(caption_files, index_files):
            # Load caption data
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            
            # Load semantic ID index
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Process this file pair
            for item in caption_data:
                video_name = item['video']
                caption = get_caption_text(item, self.dataset_name, is_train=True)
                
                # Extract video ID
                video_id = extract_video_id(video_name, self.dataset_name)
                
                # Check if this video has semantic IDs
                if video_id in index_data:
                    semantic_ids = index_data[video_id]
                    # Convert semantic_ids to string (remove < > tokens)
                    target_text = ' '.join(semantic_ids).replace("<", "").replace(">", "")
                    
                    # For ACTNET/DIDEMO, caption is a list -> create separate sample for each
                    if isinstance(caption, list):
                        for cap in caption:
                            self.source_texts.append(cap)
                            self.target_texts.append(target_text)
                    else:
                        self.source_texts.append(caption)
                        self.target_texts.append(target_text)
        
        # Apply subset sampling if specified
        if self.subset_size is not None and self.subset_size < len(self.source_texts):
            indices = list(range(len(self.source_texts)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.source_texts = [self.source_texts[i] for i in sampled_indices]
            self.target_texts = [self.target_texts[i] for i in sampled_indices]

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        if self.add_prefix:
            source_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction:{source_text} Response:"
        target_text = self.target_texts[idx]

        # Encode source with tokenizer
        source_encodings = self.tokenizer(
            source_text,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        )

        # Encode target with tokenizer
        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False
        )

        # Truncate if needed (reserve space for EOS)
        if len(target_ids) > self.max_target_len - 1:
            target_ids = target_ids[:self.max_target_len - 1]

        # Add EOS token
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        # Pad to max_length
        pad_length = self.max_target_len - len(target_ids)
        target_ids = target_ids + [self.tokenizer.pad_token_id] * pad_length

        # Convert to tensor
        labels = torch.tensor(target_ids)
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encodings["input_ids"].squeeze(),
            "attention_mask": source_encodings["attention_mask"].squeeze(),
            "labels": labels
        }


class SemanticIDDataset(Dataset):
    """
    T5 training dataset for end-to-end VideoRQVAE + T5 training.

    Semantic IDs are generated on-the-fly in the trainer using indices from
    VideoRQVAE forward pass. Dataset only handles caption tokenization and
    k-means token assignment.
    """

    def __init__(
        self,
        dataset_name,
        split,
        video_features,
        text_features,
        encoder_tokenizer,
        decoder_tokenizer,
        max_source_len=128,
        max_target_len=6,
        num_latent_tokens=4,
        code_num=64,
        codebook_layers=4,
        cache_dir='./cache',
        device='cuda',
        add_prefix=False,
        subset_size=None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.video_features = video_features
        self.text_features = text_features
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.num_latent_tokens = num_latent_tokens
        self.code_num = code_num
        self.codebook_layers = codebook_layers
        self.add_prefix = add_prefix

        # Load caption annotations
        annotations = load_caption_annotations(dataset_name, split)

        # Build samples: one per caption
        self.samples = []
        for item in annotations:
            video_name = item['video']
            video_id = video_name.replace('.mp4', '')

            # Only include videos that have features
            if video_id not in video_features:
                continue

            # Handle different dataset formats
            if dataset_name == "msrvtt":
                caption = item['caption']
            elif dataset_name == "activitynet":
                caption = "".join(item['caption']).strip()
            elif dataset_name == "didemo":
                caption = item['caption']
            elif dataset_name == "lsmdc":
                caption = item['caption']
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            self.samples.append({
                'caption': caption,
                'video_id': video_id,
            })

        # Apply subset sampling if specified (sample unique videos, one caption per video)
        if subset_size is not None:
            # Group sample indices by video_id
            video_to_indices = {}
            for idx, sample in enumerate(self.samples):
                vid = sample['video_id']
                if vid not in video_to_indices:
                    video_to_indices[vid] = []
                video_to_indices[vid].append(idx)

            # Sample subset_size unique videos
            all_video_ids = list(video_to_indices.keys())
            if subset_size < len(all_video_ids):
                sampled_video_ids = random.sample(all_video_ids, subset_size)
            else:
                sampled_video_ids = all_video_ids

            # Pick one random caption per sampled video
            sampled_indices = [
                random.choice(video_to_indices[vid]) for vid in sampled_video_ids
            ]
            self.samples = [self.samples[i] for i in sampled_indices]

        # Load or compute k-means text groupings
        import logging
        logger = logging.getLogger(__name__)
        self.text_groups = load_or_compute_kmeans_cache(
            dataset_name, split, video_features, text_features,
            num_latent_tokens, cache_dir, logger
        )

        # Build idx-to-text_key mapping for fast lookup in __getitem__
        # Use running counter per video_id for O(n) complexity instead of O(n²)
        self.idx_to_text_key = {}
        video_caption_counts = {}
        for idx, sample in enumerate(self.samples):
            video_id = sample['video_id']
            caption_idx = video_caption_counts.get(video_id, 0)
            text_key = f"{video_id}_{caption_idx}"
            self.idx_to_text_key[idx] = text_key
            video_caption_counts[video_id] = caption_idx + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        caption = sample['caption']
        video_id = sample['video_id']

        # Load video features (kept on CPU, moved to GPU in trainer)
        video_emb = self.video_features[video_id]
        if isinstance(video_emb, np.ndarray):
            video_features_cpu = torch.from_numpy(video_emb).float()
        else:
            video_features_cpu = video_emb.float().cpu() if video_emb.is_cuda else video_emb.float()

        # Tokenize caption with encoder tokenizer
        source_text = caption
        if self.add_prefix:
            source_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction:{source_text} Response:"

        source_encodings = self.encoder_tokenizer(
            source_text,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        )

        # Get k-means token assignment (used by trainer for label generation)
        text_key = self.idx_to_text_key.get(idx, f"{video_id}_0")
        token_idx = self.text_groups.get(text_key, 0)  # Default to token 0

        # Load text embedding for contrastive loss
        text_emb = self.text_features[text_key]
        if isinstance(text_emb, np.ndarray):
            text_emb_cpu = torch.from_numpy(text_emb).float()
        else:
            text_emb_cpu = text_emb.float().cpu() if text_emb.is_cuda else text_emb.float()

        return {
            "input_ids": source_encodings["input_ids"].squeeze(),
            "attention_mask": source_encodings["attention_mask"].squeeze(),
            "dataset_idx": idx,  # Dataset index for GT code lookup during evaluation
            "token_idx": token_idx,
            "video_features": video_features_cpu,  # [512] pooled InternVideo2 features
            "text_emb": text_emb_cpu  # [text_dim] text embedding for contrastive loss
        }


class SemanticIDTestDataset(Dataset):
    """
    Test dataset for end-to-end VideoRQVAE + T5 evaluation.

    Semantic IDs are generated on-the-fly during evaluation.
    Dataset only handles caption tokenization.
    """

    def __init__(
        self,
        dataset_name,
        split,
        video_features,
        encoder_tokenizer,
        decoder_tokenizer,
        max_source_len=128,
        num_latent_tokens=4,
        code_num=64,
        codebook_layers=4,
        device='cuda',
        add_prefix=False,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.video_features = video_features
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_source_len = max_source_len
        self.num_latent_tokens = num_latent_tokens
        self.code_num = code_num
        self.codebook_layers = codebook_layers
        self.add_prefix = add_prefix

        # Load caption annotations
        annotations = load_caption_annotations(dataset_name, split)

        # Build test samples (one per video)
        self.samples = []
        seen_videos = set()
        for item in annotations:
            video_name = item['video']
            video_id = video_name.replace('.mp4', '')

            # Only include videos that have features (one caption per video)
            if video_id not in video_features or video_id in seen_videos:
                continue

            seen_videos.add(video_id)

            # Handle different dataset formats
            if dataset_name == "msrvtt":
                caption = item['caption']
            elif dataset_name == "activitynet":
                caption = "".join(item['caption']).strip()
            elif dataset_name == "didemo":
                caption = item['caption']
            elif dataset_name == "lsmdc":
                caption = item['caption']
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            self.samples.append({
                'caption': caption,
                'video_id': video_id,
                'video_name': video_name
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        caption = sample['caption']
        video_id = sample['video_id']

        # Load video features (kept on CPU, moved to GPU in trainer)
        video_emb = self.video_features[video_id]
        if isinstance(video_emb, np.ndarray):
            video_features_cpu = torch.from_numpy(video_emb).float()
        else:
            video_features_cpu = video_emb.float().cpu() if video_emb.is_cuda else video_emb.float()

        # Tokenize caption
        source_text = caption
        if self.add_prefix:
            source_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction:{source_text} Response:"

        source_encodings = self.encoder_tokenizer(
            source_text,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source_encodings["input_ids"].squeeze(),
            "attention_mask": source_encodings["attention_mask"].squeeze(),
            "dataset_idx": idx,  # Dataset index for GT code lookup during evaluation
            "token_idx": 0,  # Default token_idx=0 for test dataset (no k-means routing)
            "video_id": video_id,
            "video_features": video_features_cpu  # [512] pooled InternVideo2 features
        }


@dataclass
class DataCollator:
    """Custom collator that handles both tensor and non-tensor fields.

    Pads tensor fields (input_ids, labels) and preserves non-tensor fields
    (video_id, target_texts) without attempting to pad them.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: str = "max_length"
    max_length: int = 128

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate tensor and non-tensor fields
        # Standard seq2seq keys that need padding via tokenizer
        padding_keys = {"input_ids", "labels", "attention_mask"}
        # Additional tensor keys that need stacking (not padding)
        stackable_keys = {"video_features", "text_emb"}
        # Scalar keys that need to be converted to tensors (e.g., indices)
        scalar_keys = {"dataset_idx", "token_idx"}
        non_tensor_keys = set()

        # Detect which keys are present
        if features:
            all_keys = set(features[0].keys())
            non_tensor_keys = all_keys - padding_keys - stackable_keys - scalar_keys

        # Extract non-tensor fields
        non_tensor_data = {}
        for key in non_tensor_keys:
            non_tensor_data[key] = [f.get(key) for f in features]

        # Extract stackable tensor fields (e.g., video_features)
        stackable_data = {}
        for key in stackable_keys:
            if features and key in features[0]:
                tensors = [f[key] for f in features]
                stackable_data[key] = torch.stack(tensors, dim=0)

        # Extract scalar fields and convert to tensors (e.g., dataset_idx, token_idx)
        scalar_data = {}
        for key in scalar_keys:
            if features and key in features[0]:
                values = [f[key] for f in features]
                scalar_data[key] = torch.tensor(values, dtype=torch.long)

        # Create features dict with only padding keys for tokenizer.pad()
        tensor_features = [
            {k: v for k, v in f.items() if k in padding_keys}
            for f in features
        ]

        # Pad tensor fields and convert to plain dict (BatchEncoding's update() may not work properly)
        batch = dict(self.tokenizer.pad(
            tensor_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        ))

        # Add back stackable tensor fields
        batch.update(stackable_data)

        # Add back scalar tensor fields
        batch.update(scalar_data)

        # Add back non-tensor fields
        batch.update(non_tensor_data)

        return batch
