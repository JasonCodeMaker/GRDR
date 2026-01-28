import json
import logging
import os
import pickle
from collections import defaultdict
from time import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import DataCollatorWithPadding


# Feature extractor constants
FEATURE_EXTRACTOR = "InternVideo2"
FEATURE_SUFFIX = "internvideo2"
VIDEO_SUBDIR = "video"


def set_color(log, color):
    """Add color to log strings (for terminal output)."""
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except ValueError:
        index = len(color_set) - 1
    prev_log = '\033[1;3%dm' % index + log + '\033[0m'
    return prev_log


def write_pkl(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_shared_features(dataset_name, features_root, logger, use_pseudo_queries=False):
    """
    Load InternVideo2 features shared between VideoRQVAE and T5 training.

    Args:
        dataset_name: Dataset name (msrvtt, didemo, etc.)
        features_root: Root directory for features
        logger: Logger instance or Accelerator object
        use_pseudo_queries: If True, also load and merge pseudo query text features

    Returns:
        feature_cache: Dict with keys:
            - train_video: {video_id: numpy array}
            - train_text: {text_key: numpy array}
            - test_video: {video_id: numpy array}
            - test_text: {text_key: numpy array}
    """
    # Handle logger objects, Accelerator objects, or callable (like print)
    from accelerate import Accelerator
    is_accelerator = isinstance(logger, Accelerator)
    is_callable = callable(logger) and not is_accelerator
    
    def log_info(msg):
        if is_accelerator:
            logger.print(msg)
        elif is_callable:
            logger(msg)
        else:
            logger.info(msg)
    
    def log_warning(msg):
        if is_accelerator:
            logger.print(msg)
        elif is_callable:
            logger(msg)
        else:
            logger.warning(msg)
    
    features_path = os.path.abspath(os.path.join(features_root, FEATURE_EXTRACTOR))
    if not os.path.isdir(features_path):
        raise FileNotFoundError(f"Features directory not found: {features_path}")

    log_info(set_color(f"Loading InternVideo2 features for {dataset_name.upper()}...", "green"))

    def _load_pickle(path, description, required=True):
        if not os.path.exists(path):
            message = f"{description} not found: {path}"
            if required:
                raise FileNotFoundError(message)
            log_warning(message)
            return None

        log_info(f"Loading {description} from {path}")
        start = time()
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        log_info(f"Loaded {len(data)} {description} in {time() - start:.2f}s")
        return data

    # Load video features
    train_video = _load_pickle(
        os.path.join(features_path, f"{dataset_name}/video_embeddings_train.pkl"),
        "train video features"
    )
    test_video = _load_pickle(
        os.path.join(features_path, f"{dataset_name}/video_embeddings_test.pkl"),
        "test video features",
        required=False
    )

    # Load text features
    train_text = _load_pickle(
        os.path.join(features_path, f"{dataset_name}/text_embeddings_train.pkl"),
        "train text features",
        required=False
    )
    test_text = _load_pickle(
        os.path.join(features_path, f"{dataset_name}/text_embeddings_test.pkl"),
        "test text features",
        required=False
    )

    # Merge pseudo query text features if enabled
    if use_pseudo_queries and train_text is not None:
        pseudo_path = os.path.join(features_path, f"{dataset_name}/text_embeddings_train_pesudo.pkl")
        if os.path.exists(pseudo_path):
            pseudo_text = _load_pickle(pseudo_path, "pseudo train text features", required=False)
            if pseudo_text:
                # Count original captions per video for offset calculation
                # Key format: "video0_0", "video0_1", ... -> count captions per video
                original_counts = {}
                for key in train_text.keys():
                    # Handle video IDs with underscores (e.g., "v_abc_123_0")
                    video_id = '_'.join(key.rsplit('_', 1)[:-1])
                    caption_idx = int(key.rsplit('_', 1)[-1])
                    original_counts[video_id] = max(original_counts.get(video_id, 0), caption_idx + 1)

                # Remap pseudo keys with offset: pseudo video0_0 -> merged video0_20
                # This aligns with VideoTextDataset's running counter that continues
                # from original caption count when combining annotations
                remapped = 0
                for key, emb in pseudo_text.items():
                    video_id = '_'.join(key.rsplit('_', 1)[:-1])
                    pseudo_idx = int(key.rsplit('_', 1)[-1])
                    offset = original_counts.get(video_id, 0)
                    new_key = f"{video_id}_{offset + pseudo_idx}"
                    if new_key not in train_text:  # Don't overwrite original features
                        train_text[new_key] = emb
                        remapped += 1

                log_info(f"Merged {remapped} pseudo text features with offset remapping")
        else:
            log_warning(f"Pseudo text features not found: {pseudo_path}")

    log_info(set_color("Features loaded successfully!", "green"))

    return {
        'train_video': train_video,
        'train_text': train_text,
        'test_video': test_video if test_video else {},
        'test_text': test_text if test_text else {},
    }


def load_or_compute_kmeans_cache(dataset_name, split, video_features, text_features,
                                   num_latent_tokens, cache_dir, logger=None,
                                   use_pseudo_queries=False):
    """
    Load k-means text groupings from cache or compute fresh.

    Args:
        dataset_name: Dataset name
        split: 'train' or 'test'
        video_features: Dict of video embeddings
        text_features: Dict of text embeddings
        num_latent_tokens: Number of latent tokens (k for k-means)
        cache_dir: Cache directory
        logger: Logger instance
        use_pseudo_queries: If True, use separate cache for pseudo-enabled mode

    Returns:
        text_groups: {text_key: assigned_token_idx}
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Cache path with pseudo suffix when applicable
    pseudo_suffix = "_pseudo" if use_pseudo_queries else ""
    cache_path = os.path.join(
        cache_dir, f"{dataset_name}/{split}_kmeans_k{num_latent_tokens}{pseudo_suffix}.pkl"
    )

    # Try loading from cache
    if os.path.exists(cache_path):
        logger.info(f"Loading k-means cache from {cache_path}")
        with open(cache_path, 'rb') as f:
            text_groups = pickle.load(f)
        logger.info(f"Loaded {len(text_groups)} text group assignments from cache")
        return text_groups

    # Compute fresh
    logger.info(f"Computing k-means text groupings for {split} split...")
    text_groups = compute_kmeans_groupings(
        video_features, text_features, num_latent_tokens, logger
    )

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(text_groups, f)
    logger.info(f"Saved k-means cache to {cache_path}")

    return text_groups


def compute_kmeans_groupings(video_features, text_features, num_latent_tokens, logger=None):
    """
    Cluster captions using k-means for multi-text VideoRQVAE training.

    For each video with multiple captions, cluster captions into
    num_latent_tokens groups. Each caption is assigned to a token index.

    Args:
        video_features: Dict of video embeddings
        text_features: Dict of text embeddings
        num_latent_tokens: Number of clusters (k)
        logger: Logger instance

    Returns:
        text_groups: {text_key: token_idx} mapping
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    text_groups = {}

    # Group texts by video
    video_text_map = defaultdict(list)
    for text_key in text_features.keys():
        # Extract video ID from text key (e.g., "video0_0" -> "video0")
        video_id = '_'.join(text_key.split('_')[:-1])
        if video_id in video_features:
            video_text_map[video_id].append(text_key)

    logger.info(f"Clustering captions for {len(video_text_map)} videos...")

    # K-means clustering per video
    for video_id, text_keys in tqdm(video_text_map.items(), desc="K-means clustering"):
        # Get text embeddings
        text_embs = np.array([text_features[tk] for tk in text_keys])
        num_texts = len(text_keys)
        k = min(num_latent_tokens, num_texts)

        if k == 1:
            # All captions use first token
            labels = np.zeros(num_texts, dtype=int)
        else:
            # K-means clustering
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(text_embs)
            except Exception as e:
                logger.warning(f"K-means failed for {video_id}: {e}, assigning all to token 0")
                labels = np.zeros(num_texts, dtype=int)

        # Assign token indices
        for text_key, label in zip(text_keys, labels):
            text_groups[text_key] = int(label)

    logger.info(f"K-means clustering complete: {len(text_groups)} text assignments")

    return text_groups

