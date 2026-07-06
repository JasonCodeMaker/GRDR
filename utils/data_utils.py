import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from time import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from utils.memmap_dict import MappingUnion, NpyMemmapMapping


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


def kmeans_cache_path(dataset_name, split, num_latent_tokens, cache_dir,
                      use_pseudo_queries=False):
    pseudo_suffix = "_pseudo" if use_pseudo_queries else ""
    return os.path.join(
        cache_dir, f"{dataset_name}/{split}_kmeans_k{num_latent_tokens}{pseudo_suffix}.pkl"
    )


def has_kmeans_cache(dataset_name, split, num_latent_tokens, cache_dir,
                     use_pseudo_queries=False):
    return os.path.exists(
        kmeans_cache_path(dataset_name, split, num_latent_tokens, cache_dir,
                          use_pseudo_queries=use_pseudo_queries)
    )


def _compute_original_caption_counts(train_text):
    original_counts = {}
    for key in train_text.keys():
        video_id, suffix = key.rsplit('_', 1)
        if suffix.startswith('a'):
            continue
        original_counts[video_id] = max(
            original_counts.get(video_id, 0), int(suffix) + 1
        )
    return original_counts


def _remap_pseudo_key(key, original_counts):
    video_id, suffix = key.rsplit('_', 1)
    idx_str = suffix[1:] if suffix.startswith('a') else suffix
    pseudo_idx = int(idx_str)
    offset = original_counts.get(video_id, 0)
    return f"{video_id}_{offset + pseudo_idx}"


def _file_size_if_exists(path):
    return os.path.getsize(path) if os.path.exists(path) else None


def _pseudo_memmap_paths(pseudo_pickle_path):
    base = os.path.splitext(pseudo_pickle_path)[0]
    npy_path = f"{base}.npy"
    return npy_path, f"{base}.idx.json", f"{npy_path}.meta.json"


def _load_pseudo_memmap(pseudo_pickle_path, train_text_path, log_info, log_warning,
                        strict=False):
    npy_path, idx_path, meta_path = _pseudo_memmap_paths(pseudo_pickle_path)
    required = (npy_path, idx_path, meta_path)
    if not all(os.path.exists(p) for p in required):
        return None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    except Exception as exc:
        message = f"Could not read pseudo memmap metadata {meta_path}: {exc}"
        if strict:
            raise RuntimeError(message) from exc
        log_warning(f"{message}; falling back to legacy pickle")
        return None

    if meta.get('key_mode') != 'remapped':
        message = f"Pseudo memmap has unsupported key_mode={meta.get('key_mode')}"
        if strict:
            raise RuntimeError(message)
        log_warning(f"{message}; falling back to legacy pickle")
        return None

    sources = meta.get('sources', {})
    expected_pseudo_size = sources.get('pseudo_pickle', {}).get('size')
    actual_pseudo_size = _file_size_if_exists(pseudo_pickle_path)
    if actual_pseudo_size is not None and expected_pseudo_size != actual_pseudo_size:
        message = (
            f"Pseudo memmap source size mismatch for {pseudo_pickle_path}: "
            f"meta={expected_pseudo_size} actual={actual_pseudo_size}"
        )
        if strict:
            raise RuntimeError(message)
        log_warning(f"{message}; falling back to legacy pickle")
        return None

    expected_train_size = sources.get('train_text_pickle', {}).get('size')
    actual_train_size = _file_size_if_exists(train_text_path)
    if actual_train_size is not None and expected_train_size != actual_train_size:
        message = (
            f"Pseudo memmap train-text source size mismatch for {train_text_path}: "
            f"meta={expected_train_size} actual={actual_train_size}"
        )
        if strict:
            raise RuntimeError(message)
        log_warning(f"{message}; falling back to legacy pickle")
        return None

    start = time()
    mapping = NpyMemmapMapping(npy_path, idx_path)
    log_info(
        f"Loaded {len(mapping)} pseudo train text features from memmap "
        f"{npy_path} in {time() - start:.2f}s"
    )
    return mapping


def load_shared_features(dataset_name, features_root, logger, use_pseudo_queries=False,
                         load_train_video=True, load_train_text=True,
                         load_test_video=True, load_test_text=True,
                         prefer_memmap=True, strict_memmap=False):
    """
    Load InternVideo2 features shared between VideoRQVAE and T5 training.

    Args:
        dataset_name: Dataset name (msrvtt, didemo, etc.)
        features_root: Root directory for features
        logger: Logger instance or Accelerator object
        use_pseudo_queries: If True, also load and merge pseudo query text features
        load_*: Allow callers to skip feature stores that are not needed.
        prefer_memmap: Use converted pseudo-query memmap artifacts when available.
        strict_memmap: Raise instead of falling back when memmap artifacts are stale.

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

    dataset_dir = os.path.join(features_path, dataset_name)
    train_video_path = os.path.join(dataset_dir, "video_embeddings_train.pkl")
    test_video_path = os.path.join(dataset_dir, "video_embeddings_test.pkl")
    train_text_path = os.path.join(dataset_dir, "text_embeddings_train.pkl")
    test_text_path = os.path.join(dataset_dir, "text_embeddings_test.pkl")
    pseudo_path = os.path.join(dataset_dir, "text_embeddings_train_addition.pkl")

    load_jobs = {}
    skipped = {
        'train_video': not load_train_video,
        'test_video': not load_test_video,
        'train_text': not load_train_text,
        'test_text': not load_test_text,
    }

    def _add_job(name, should_load, path, description, required=True):
        if should_load:
            load_jobs[name] = (path, description, required)
        else:
            log_info(f"Skipping {description} load by request")

    _add_job('train_video', load_train_video, train_video_path, "train video features")
    _add_job('test_video', load_test_video, test_video_path, "test video features",
             required=False)
    _add_job('train_text', load_train_text, train_text_path, "train text features",
             required=False)
    _add_job('test_text', load_test_text, test_text_path, "test text features",
             required=False)

    pseudo_job = None
    if use_pseudo_queries and load_train_text:
        if prefer_memmap:
            pseudo_memmap = _load_pseudo_memmap(
                pseudo_path, train_text_path, log_info, log_warning,
                strict=strict_memmap,
            )
            if pseudo_memmap is not None:
                pseudo_job = ('memmap', pseudo_memmap)
            elif os.path.exists(pseudo_path):
                load_jobs['pseudo_text'] = (
                    pseudo_path, "pseudo train text features", False
                )
                pseudo_job = ('pickle', None)
        elif os.path.exists(pseudo_path):
            load_jobs['pseudo_text'] = (
                pseudo_path, "pseudo train text features", False
            )
            pseudo_job = ('pickle', None)
    elif use_pseudo_queries and not load_train_text:
        log_info("Skipping pseudo train text features because train text load is skipped")

    loaded = {}
    if load_jobs:
        max_workers = min(5, len(load_jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                name: pool.submit(_load_pickle, path, description, required)
                for name, (path, description, required) in load_jobs.items()
            }
            for name, future in futures.items():
                loaded[name] = future.result()

    train_video = loaded.get('train_video') if not skipped['train_video'] else {}
    test_video = loaded.get('test_video') if not skipped['test_video'] else {}
    train_text = loaded.get('train_text') if not skipped['train_text'] else {}
    test_text = loaded.get('test_text') if not skipped['test_text'] else {}

    # Merge pseudo query text features if enabled (unified file shared with AVG/MM-SemanticTVR).
    # The addition pickle uses `<vid>_a<idx>` keys; we remap them into the running `<vid>_<int>`
    # scheme that VideoTextDataset's counter produces when --use_pseudo_queries is on.
    if use_pseudo_queries and load_train_text and train_text is not None:
        if pseudo_job is not None and pseudo_job[0] == 'memmap':
            train_text = MappingUnion(train_text, pseudo_job[1], assume_disjoint=True)
            log_info(f"Unioned train text with {len(pseudo_job[1])} memmap pseudo features")
        elif pseudo_job is not None and pseudo_job[0] == 'pickle':
            pseudo_text = loaded.get('pseudo_text')
            if pseudo_text:
                original_counts = _compute_original_caption_counts(train_text)
                remapped = 0
                for key, emb in pseudo_text.items():
                    new_key = _remap_pseudo_key(key, original_counts)
                    if new_key not in train_text:
                        train_text[new_key] = emb
                        remapped += 1

                log_info(f"Merged {remapped} addition text features with offset remapping")
        else:
            log_warning(f"Addition text features not found: {pseudo_path}")

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

    cache_path = kmeans_cache_path(
        dataset_name, split, num_latent_tokens, cache_dir,
        use_pseudo_queries=use_pseudo_queries
    )

    # Try loading from cache
    if os.path.exists(cache_path):
        logger.info(f"Loading k-means cache from {cache_path}")
        with open(cache_path, 'rb') as f:
            text_groups = pickle.load(f)
        logger.info(f"Loaded {len(text_groups)} text group assignments from cache")
        return text_groups

    # Compute fresh
    if not text_features:
        raise RuntimeError(
            f"K-means cache missing at {cache_path}, but text features were not "
            "loaded. Disable cache-aware text skipping or build the cache first."
        )

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
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=20, algorithm='lloyd')
                labels = kmeans.fit_predict(text_embs)
            except Exception as e:
                logger.warning(f"K-means failed for {video_id}: {e}, assigning all to token 0")
                labels = np.zeros(num_texts, dtype=int)

        # Assign token indices
        for text_key, label in zip(text_keys, labels):
            text_groups[text_key] = int(label)

    logger.info(f"K-means clustering complete: {len(text_groups)} text assignments")

    return text_groups
