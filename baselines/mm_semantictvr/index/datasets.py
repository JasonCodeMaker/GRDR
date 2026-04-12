import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import logging
from time import time

# Dataset directory name mapping for InternVideo2 features
# Maps common dataset names to their actual directory names
DATASET_DIR_MAP = {
    'msrvtt': 'msrvtt',
    'activitynet': 'actnet',
    'actnet': 'actnet',
    'didemo': 'didemo',
    'lsmdc': 'lsmdc'
}

def get_dataset_dir(dataset_name: str) -> str:
    """Get the actual directory name for a dataset.

    Args:
        dataset_name: Dataset name (e.g., 'msrvtt', 'activitynet', 'actnet')

    Returns:
        Actual directory name used in the features path
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_DIR_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_DIR_MAP.keys())}")
    return DATASET_DIR_MAP[dataset_name]


def load_internvideo2_features(dataset_name: str, features_root: str):
    """Load InternVideo2 video and text features for any supported dataset.

    Features are stored in the structure:
        {features_root}/InternVideo2/{dataset_dir}/video_embeddings_{split}.pkl
        {features_root}/InternVideo2/{dataset_dir}/text_embeddings_{split}.pkl

    Args:
        dataset_name: Dataset name (e.g., 'msrvtt', 'actnet', 'didemo', 'lsmdc')
        features_root: Root path to features directory

    Returns:
        Tuple of (train_video, train_text, test_video, test_text) dictionaries.
        Text/test features may be None if not available.
    """
    logger = logging.getLogger(__name__)

    # Normalize dataset name to directory name
    dataset_dir = get_dataset_dir(dataset_name)
    base_path = os.path.abspath(os.path.join(features_root, "InternVideo2", dataset_dir))

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Features directory not found: {base_path}")

    logger.info(f"Loading InternVideo2 features for {dataset_name.upper()} from {base_path}")

    def _load_pickle(filename: str, description: str, required: bool = True):
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            message = f"{description} not found: {path}"
            if required:
                raise FileNotFoundError(message)
            logger.warning(message)
            return None

        logger.info(f"Loading {description} from {path}")
        start = time()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data)} {description} in {time() - start:.2f}s")
        return data

    train_video = _load_pickle("video_embeddings_train.pkl", "train video features")
    train_text = _load_pickle("text_embeddings_train.pkl", "train text features", required=False)
    test_video = _load_pickle("video_embeddings_test.pkl", "test video features", required=False)
    test_text = _load_pickle("text_embeddings_test.pkl", "test text features", required=False)

    return train_video, train_text, test_video, test_text


class MultiTextVideoDataset(data.Dataset):
    """Multi-text dataset for VideoRQVAE training with all captions per video

    Optimized for MSRVTT where each video has exactly 20 captions.
    Returns one video with all its corresponding text embeddings for
    efficient multi-positive learning and router training.
    """

    def __init__(self, dataset_name, features_root="./dataset/features", split="train",
                 feature_extractor="CLIP", video_features=None, text_features=None, num_latent_tokens=1):
        """
        Args:
            dataset_name: 'msrvtt', 'didemo', 'activitynet', or 'lsmdc'
            features_root: Root path to features directory
            split: 'train' or 'test'
            feature_extractor: 'CLIP' or 'InternVL' - determines feature subdirectory
            video_features: Pre-loaded video features dict (optional, for efficiency)
            text_features: Pre-loaded text features dict (optional, for efficiency)
            num_latent_tokens: Number of latent tokens for text grouping (default: 1)
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.feature_extractor = feature_extractor
        self.num_latent_tokens = num_latent_tokens

        # Set features path based on feature_extractor
        if feature_extractor == "CLIP":
            features_path = os.path.join(features_root, "CLIP")
            feature_suffix = "cliplargel14"
        elif feature_extractor == "InternVL":
            features_path = os.path.join(features_root, "InternVL")
            feature_suffix = "internvl-hico-r16"
        elif feature_extractor == "InternVideo2":
            # InternVideo2 uses per-dataset subdirectories
            dataset_dir = get_dataset_dir(self.dataset_name)
            features_path = os.path.join(features_root, "InternVideo2", dataset_dir)
            feature_suffix = None  # InternVideo2 doesn't use suffix in filenames
        else:
            raise ValueError(f"Unsupported feature_extractor: {feature_extractor}. Must be 'CLIP', 'InternVL', or 'InternVideo2'")

        self.features_root = os.path.abspath(features_path)

        # Load video features
        if video_features is not None:
            self.video_features = video_features
            print(f"Using pre-loaded video features: {len(video_features)} samples")
        else:
            if feature_suffix:
                video_path = os.path.join(self.features_root, f"{self.dataset_name}_{feature_suffix}_video_embeddings_{self.split}.pkl")
            else:
                video_path = os.path.join(self.features_root, f"video_embeddings_{self.split}.pkl")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video features not found: {video_path}")
            with open(video_path, 'rb') as f:
                self.video_features = pickle.load(f)

        # Load text features - required for multi-text training
        if text_features is not None:
            self.text_features = text_features
            print(f"Using pre-loaded text features: {len(text_features)} samples")
        else:
            if feature_suffix:
                text_path = os.path.join(self.features_root, f"{self.dataset_name}_{feature_suffix}_text_embeddings_{self.split}.pkl")
            else:
                text_path = os.path.join(self.features_root, f"text_embeddings_{self.split}.pkl")
            if not os.path.exists(text_path):
                raise FileNotFoundError(f"Text features not found: {text_path}")
            with open(text_path, 'rb') as f:
                self.text_features = pickle.load(f)

        # Group texts by video for multi-text training
        self._create_multi_text_data()

        print(f"Loaded {len(self.video_ids)} videos with multi-text data for {self.dataset_name}")
        if len(self.video_ids) > 0:
            sample_video = self.video_text_groups[self.video_ids[0]]['video']
            sample_texts = self.video_text_groups[self.video_ids[0]]['texts']
            print(f"Video shape: {sample_video.shape}, Texts per video: {len(sample_texts)}, Text shape: {sample_texts[0].shape}")

    def _extract_video_key_from_text(self, text_key):
        """Extract video key from text key.

        All datasets use the format: {video_key}_{caption_index}
        Using rsplit ensures correct handling of keys with multiple underscores (e.g., LSMDC).

        Examples:
            MSRVTT: 'video1234_0' -> 'video1234'
            ACTNET: 'v_QOlSCBRmfWY_0' -> 'v_QOlSCBRmfWY'
            DiDeMo: '54322086@N00_2408598493_274c77d26a_0' -> '54322086@N00_2408598493_274c77d26a'
            LSMDC: '3001_21_JUMP_STREET_00.02.55.644-00.02.56.718_0' -> '3001_21_JUMP_STREET_00.02.55.644-00.02.56.718'
        """
        return text_key.rsplit('_', 1)[0]

    def _create_multi_text_data(self):
        """Group text embeddings by video for multi-text training.

        Supports all datasets: msrvtt, actnet, didemo, lsmdc
        """
        self.video_text_groups = {}

        # Load or compute text group mapping cache (only for InternVideo2)
        text_group_mapping_cache = None
        if self.feature_extractor == "InternVideo2":
            text_group_mapping_cache = self._load_or_compute_text_group_mapping()

        # Group texts by video base key (works for all datasets)
        text_groups = {}
        for text_key in self.text_features.keys():
            video_key = self._extract_video_key_from_text(text_key)
            if video_key not in text_groups:
                text_groups[video_key] = []
            text_groups[video_key].append(text_key)

        # Create video-text groups for videos that have both video and text features
        video_keys = list(self.video_features.keys())

        for video_key in video_keys:
            if video_key in text_groups:
                text_keys = sorted(text_groups[video_key])  # Sort for consistency
                text_embeddings = [self.text_features[tk] for tk in text_keys]

                self.video_text_groups[video_key] = {
                    'video': self.video_features[video_key],
                    'texts': text_embeddings,
                    'text_keys': text_keys
                }

                # Apply text grouping for InternVideo2 using cache
                if self.feature_extractor == "InternVideo2" and text_group_mapping_cache is not None:
                    if video_key in text_group_mapping_cache:
                        group_ids = text_group_mapping_cache[video_key]
                        self.video_text_groups[video_key]['text_group_ids'] = group_ids

        self.video_ids = list(self.video_text_groups.keys())

        # Validate consistent text count (especially for MSRVTT)
        if self.dataset_name == 'msrvtt' and self.video_ids:
            text_counts = [len(self.video_text_groups[vid]['texts']) for vid in self.video_ids]
            if not all(count == text_counts[0] for count in text_counts):
                print(f"Warning: Inconsistent text counts per video: {set(text_counts)}")
            else:
                print(f"Consistent {text_counts[0]} texts per video for {self.dataset_name}")

    def _load_or_compute_text_group_mapping(self):
        """
        Load cached text group mapping or compute and cache it.
        Only used for InternVideo2 feature extractor.

        Cache file: {dataset}_internvideo2_text_group_mapping_{split}_k{num_latent_tokens}.pkl

        Returns:
            dict: Mapping from video_key -> group_ids (numpy array)
        """
        cache_filename = f"{self.dataset_name}_internvideo2_text_group_mapping_{self.split}_k{self.num_latent_tokens}.pkl"
        cache_path = os.path.join(self.features_root, cache_filename)

        # Try loading from cache
        if os.path.exists(cache_path):
            print(f"Loading text group mapping from cache: {cache_filename}")
            with open(cache_path, 'rb') as f:
                text_group_mapping = pickle.load(f)
            print(f"Loaded {len(text_group_mapping)} video text group mappings from cache")
            return text_group_mapping

        # Cache doesn't exist - compute the mapping
        print(f"Computing text group mapping for k={self.num_latent_tokens}...")
        text_group_mapping = {}

        # Group texts by video base key (using unified extraction)
        text_groups = {}
        for text_key in self.text_features.keys():
            video_key = self._extract_video_key_from_text(text_key)
            if video_key not in text_groups:
                text_groups[video_key] = []
            text_groups[video_key].append(text_key)

        # Compute group IDs for each video
        video_keys = list(self.video_features.keys())
        for i, video_key in enumerate(video_keys):
            if video_key in text_groups:
                text_keys = sorted(text_groups[video_key])
                text_embeddings = [self.text_features[tk] for tk in text_keys]

                group_ids = self.group_text_embs(
                    self.video_features[video_key],
                    text_embeddings
                )
                text_group_mapping[video_key] = group_ids

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(video_keys)} videos")

        # Save to cache
        print(f"Saving text group mapping to cache: {cache_filename}")
        with open(cache_path, 'wb') as f:
            pickle.dump(text_group_mapping, f)
        print(f"Cached {len(text_group_mapping)} video text group mappings")

        return text_group_mapping

    def group_text_embs(self, video_emb, text_embs):
        """
        Groups text embeddings using k-means and assigns group IDs based on
        video-text similarity.

        Args:
            video_emb: Video embedding tensor [dim]
            text_embs: List of text embeddings, each [dim]

        Returns:
            group_ids: np.ndarray of group IDs for each text [num_texts]
        """
        from sklearn.cluster import KMeans
        import torch.nn.functional as F

        # Handle edge case: if num_latent_tokens == 1, all texts belong to group 0
        if self.num_latent_tokens == 1:
            return np.zeros(len(text_embs), dtype=np.int64)

        # Stack text embeddings into tensor [num_texts, dim]
        if isinstance(text_embs[0], np.ndarray):
            text_tensor = torch.from_numpy(np.stack(text_embs))
        else:
            text_tensor = torch.stack(text_embs)

        # Convert video_emb to tensor
        if isinstance(video_emb, np.ndarray):
            video_tensor = torch.from_numpy(video_emb)
        else:
            video_tensor = video_emb

        num_texts = text_tensor.shape[0]
        k = min(self.num_latent_tokens, num_texts)  # Handle case where k > num_texts

        # Apply k-means clustering on text embeddings
        text_np = text_tensor.cpu().numpy() if isinstance(text_tensor, torch.Tensor) else text_tensor
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(text_np)

        # Compute average cosine similarity for each cluster with video
        cluster_similarities = []
        for cluster_id in range(k):
            # Get texts in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = text_tensor[cluster_mask]

            # Compute cosine similarity between video and each text in cluster
            video_norm = F.normalize(video_tensor.unsqueeze(0).float(), p=2, dim=-1)
            cluster_norm = F.normalize(cluster_texts.float(), p=2, dim=-1)
            similarities = (video_norm * cluster_norm).sum(dim=-1)

            # Average similarity for this cluster
            avg_sim = similarities.mean().item()
            cluster_similarities.append(avg_sim)

        # Sort clusters by similarity (descending) and create mapping
        sorted_clusters = np.argsort(cluster_similarities)[::-1]
        cluster_to_group = {old_id: new_id for new_id, old_id in enumerate(sorted_clusters)}

        # Map cluster labels to group IDs
        group_ids = np.array([cluster_to_group[label] for label in cluster_labels], dtype=np.int64)

        return group_ids

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_text_group = self.video_text_groups[video_id]

        # Format video features based on feature extractor
        video_features = video_text_group['video']
        if isinstance(video_features, np.ndarray):
            video_features = torch.from_numpy(video_features)
        elif not isinstance(video_features, torch.Tensor):
            video_features = torch.tensor(video_features)

        # InternVideo2 uses single pooled embeddings [512] for VideoRQVAE_V2
        # CLIP/InternVL use patch embeddings [num_patches, dim] for VideoRQVAE
        if self.feature_extractor == "InternVideo2":
            # Keep as single embedding [in_dim] for VideoRQVAE_V2
            if video_features.dim() > 1:
                video_features = video_features.squeeze()
        else:
            # Ensure patch format for original VideoRQVAE
            if video_features.dim() == 1:
                video_features = video_features.reshape(1, -1)
        video_features = video_features.float().detach()

        # Stack all text embeddings for this video
        text_embeddings = []
        for text_emb in video_text_group['texts']:
            if isinstance(text_emb, np.ndarray):
                text_emb = torch.from_numpy(text_emb)
            elif not isinstance(text_emb, torch.Tensor):
                text_emb = torch.tensor(text_emb)
            text_embeddings.append(text_emb.float().detach())

        # Stack to create [num_texts, text_dim] tensor
        text_embeddings = torch.stack(text_embeddings, dim=0)

        # Get text group IDs if available (InternVideo2 features)
        text_group_ids = video_text_group.get('text_group_ids', None)
        if text_group_ids is not None:
            text_group_ids = torch.from_numpy(text_group_ids).long()

        return {
            'video_patches': video_features,        # [num_patches, video_dim]
            'text_embs': text_embeddings,           # [num_texts, text_dim]
            'video_id': video_id,
            'text_keys': video_text_group['text_keys'],
            'text_group_ids': text_group_ids       # [num_texts] or None
        }

    def __len__(self):
        return len(self.video_ids)


class VideoTextGuidedDataset(data.Dataset):
    """Unified dataset for both standard and text-guided video RQ-VAE training across multiple datasets

    Smart dataset that adapts output format based on target model:
    - RQVAE: Returns averaged single embeddings [feature_dim]
    - VideoRQVAE: Returns patch sequences [num_patches, feature_dim]

    This enables both architectures to work with the same InternVL features.
    """
    
    def __init__(self, dataset_name, features_root="./dataset/features", split="train", text_guided=True, feature_extractor="CLIP", model_type="rqvae", video_features=None, text_features=None):
        """
        Args:
            dataset_name: 'msrvtt', 'didemo', 'activitynet', or 'lsmdc'
            features_root: Root path to features directory
            split: 'train' or 'test'
            text_guided: If True, returns dict with video+text embeddings. If False, returns just video tensor.
            feature_extractor: 'CLIP' or 'InternVL' - determines feature subdirectory
            model_type: 'rqvae' (single embeddings) or 'videorqvae' (patch sequences) - determines output format
            video_features: Pre-loaded video features dict (optional, for efficiency)
            text_features: Pre-loaded text features dict (optional, for efficiency)
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.text_guided = text_guided
        self.feature_extractor = feature_extractor
        self.model_type = model_type.lower()

        # Validate model_type
        if self.model_type not in ['rqvae', 'videorqvae']:
            raise ValueError(f"model_type must be 'rqvae' or 'videorqvae', got: {model_type}")

        # Set features path and suffix based on feature_extractor
        if feature_extractor == "CLIP":
            features_path = os.path.join(features_root, "CLIP")
            self.feature_suffix = "cliplargel14"
        elif feature_extractor == "InternVL":
            features_path = os.path.join(features_root, "InternVL")
            self.feature_suffix = "internvl-hico-r16"
        elif feature_extractor == "InternVideo2":
            # InternVideo2 uses per-dataset subdirectories with simpler filenames
            dataset_dir = get_dataset_dir(self.dataset_name)
            features_path = os.path.join(features_root, "InternVideo2", dataset_dir)
            self.feature_suffix = None  # InternVideo2 doesn't use suffix in filenames
        else:
            raise ValueError(f"Unsupported feature_extractor: {feature_extractor}. Must be 'CLIP', 'InternVL', or 'InternVideo2'")

        # Convert relative path to absolute path to avoid path resolution issues
        self.features_root = os.path.abspath(features_path)

        # Load Dataset embeddings - use pre-loaded features if provided, otherwise load from files
        if video_features is not None:
            # Use pre-loaded video features for efficiency
            self.video_features = video_features
            print(f"Using pre-loaded video features: {len(video_features)} samples")
        else:
            # Build video path based on feature extractor
            if self.feature_suffix:
                video_path = os.path.join(self.features_root, f"{self.dataset_name}_{self.feature_suffix}_video_embeddings_{self.split}.pkl")
            else:
                video_path = os.path.join(self.features_root, f"video_embeddings_{self.split}.pkl")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video features not found: {video_path}")
            with open(video_path, 'rb') as f:
                self.video_features = pickle.load(f)

        # Load text embeddings - use pre-loaded features if provided, otherwise load from files
        if text_features is not None:
            # Use pre-loaded text features for efficiency
            self.text_features = text_features
            if text_features:
                print(f"Using pre-loaded text features: {len(text_features)} samples")
        else:
            # Fallback to loading from files
            self.text_features = None
            if self.text_guided:
                if self.feature_suffix:
                    text_path = os.path.join(self.features_root, f"{self.dataset_name}_{self.feature_suffix}_text_embeddings_{self.split}.pkl")
                else:
                    text_path = os.path.join(self.features_root, f"text_embeddings_{self.split}.pkl")
                if not os.path.exists(text_path):
                    raise FileNotFoundError(f"Text features not found: {text_path}")
                with open(text_path, 'rb') as f:
                    self.text_features = pickle.load(f)
        
        # Create video-text pairs or just video keys based on mode
        self.pairs = []
        self._create_data_pairs()
        
        mode_str = "text-guided pairs" if self.text_guided else "video samples"
        print(f"Loaded {len(self.pairs)} {mode_str} for {self.dataset_name}")
        if len(self.pairs) > 0:
            if self.text_guided:
                sample_video = self.video_features[self.pairs[0][0]]
                sample_text = self.text_features[self.pairs[0][1]]
                print(f"Video shape: {sample_video.shape}, Text shape: {sample_text.shape}")
            else:
                sample_video = self.video_features[self.pairs[0]]
                print(f"Video shape: {sample_video.shape}")

            # Set dimensions based on model type and feature structure
            if len(sample_video.shape) == 1:
                # Already averaged embeddings: [feature_dim]
                self.num_patches = 1
                self.dim = sample_video.shape[0]
                print(f"Single embedding mode - Feature dim: {self.dim}")
            else:
                # Patch embeddings: [num_patches, feature_dim]
                self.num_patches = sample_video.shape[0]
                self.dim = sample_video.shape[-1]
                print(f"Patch embedding mode - Patches: {self.num_patches}, Feature dim: {self.dim}")

            print(f"Model type: {self.model_type} - Will {'preserve patches' if self.model_type == 'videorqvae' else 'average patches'}")
        else:
            raise ValueError(f"No valid {mode_str} found for {self.dataset_name}")
    
    def _extract_video_key(self, text_key):
        """Extract video key from text key.

        All datasets use the format: {video_key}_{caption_index}
        Using rsplit ensures correct handling of keys with multiple underscores (e.g., LSMDC).

        Examples:
            MSRVTT: 'video1234_0' -> 'video1234'
            ACTNET: 'v_QOlSCBRmfWY_0' -> 'v_QOlSCBRmfWY'
            DiDeMo: '54322086@N00_2408598493_274c77d26a_0' -> '54322086@N00_2408598493_274c77d26a'
            LSMDC: '3001_21_JUMP_STREET_00.02.55.644-00.02.56.718_0' -> '3001_21_JUMP_STREET_00.02.55.644-00.02.56.718'
        """
        return text_key.rsplit('_', 1)[0]

    def _create_data_pairs(self):
        """Create video-text pairs or video keys based on training mode.

        Supports all datasets: msrvtt, actnet, didemo, lsmdc
        """
        if self.text_guided:
            # Create video-text pairs using unified key extraction
            video_keys = set(self.video_features.keys())
            for text_key in self.text_features.keys():
                video_key = self._extract_video_key(text_key)
                if video_key in video_keys:
                    self.pairs.append((video_key, text_key))
        else:
            # Just use video keys for standard training
            self.pairs = list(self.video_features.keys())
    
    def __getitem__(self, index):
        if self.text_guided:
            # Text-guided mode: return dict with video + text embeddings
            video_key, text_key = self.pairs[index]

            # Get video embeddings with model-appropriate format
            video_features = self.video_features[video_key]
            video_tensor = self._format_video_features(video_features)

            # Get text embedding
            text_emb = self.text_features[text_key]

            # Use appropriate key name based on model type
            video_key_name = 'video_patches' if self.model_type == 'videorqvae' else 'video_emb'

            return {
                video_key_name: video_tensor,
                'text_emb': torch.FloatTensor(text_emb).detach(),
                'video_key': video_key,
                'text_key': text_key
            }
        else:
            # Standard mode: return video data with model-appropriate format
            video_key = self.pairs[index]

            video_features = self.video_features[video_key]
            video_tensor = self._format_video_features(video_features)

            # For test split, preserve video ID for semantic ID mapping
            if self.split == "test":
                return {
                    'video_patches': video_tensor,
                    'video_id': video_key
                }
            else:
                # Backward compatibility for train split standard mode
                return video_tensor

    def _format_video_features(self, video_features):
        """
        Format video features based on model type with validation.

        Args:
            video_features: Raw video features from file (may be patches or single embedding)

        Returns:
            torch.FloatTensor: Formatted for target model type
        """
        # Input validation
        if video_features is None:
            raise ValueError("video_features cannot be None")

        import numpy as np
        if isinstance(video_features, np.ndarray):
            video_features = torch.from_numpy(video_features)
        elif not isinstance(video_features, torch.Tensor):
            video_features = torch.tensor(video_features)

        # Shape validation
        if video_features.dim() == 0:
            raise ValueError(f"Invalid video features shape: {video_features.shape}. Expected at least 1D.")

        if self.model_type == 'videorqvae':
            return video_features.float().detach()  

        elif self.model_type == 'rqvae':
            # Original RQVAE: single averaged embedding
            if video_features.dim() > 1:
                # Average patches to single embedding
                video_features = video_features.mean(dim=0)  # [feature_dim]

            # Validate feature dimension
            feature_dim = video_features.shape[0]
            if feature_dim < 1 or feature_dim > 10000:
                raise ValueError(f"Unreasonable feature dimension: {feature_dim}. Expected 1-10000.")

            return video_features.float().detach()  # [feature_dim]

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Expected 'rqvae' or 'videorqvae'.")
    
    def __len__(self):
        return len(self.pairs)
