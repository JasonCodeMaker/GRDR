"""
Utility script to pre-extract and cache video features for fast per-query evaluation.
"""

import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from datasets.video_capture import VideoCapture
from datasets.model_transforms import init_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset
from datasets.msvd_dataset import MSVDDataset
from datasets.lsmdc_dataset import LSMDCDataset
from datasets.actnet_dataset import ActivityNetDataset
from datasets.didemo_dataset import DiDeMoDataset


def get_unique_video_ids(config, split_type='test'):
    """
    Efficiently get unique video IDs directly from dataset without iterating data_loader.
    This is much faster than iterating through the entire data_loader.
    
    Args:
        config: Configuration object
        split_type: 'train' or 'test'
        
    Returns:
        List of unique video IDs
    """
    img_transforms = init_transform_dict(config.input_res)
    test_img_tfms = img_transforms['clip_test']
    
    if config.dataset_name == "MSRVTT":
        dataset = MSRVTTDataset(config, split_type, test_img_tfms)
        if split_type == 'train':
            return list(dataset.train_vids)
        else:
            return dataset.test_df['video_id'].unique().tolist()
            
    elif config.dataset_name == "MSVD":
        dataset = MSVDDataset(config, split_type, test_img_tfms)
        if split_type == 'train':
            return list(dataset.train_vids)
        else:
            return dataset.test_df['video_id'].unique().tolist()
            
    elif config.dataset_name == 'LSMDC':
        dataset = LSMDCDataset(config, split_type, test_img_tfms)
        # LSMDC uses clip2caption dict for both train and test
        return list(dataset.clip2caption.keys())
            
    elif config.dataset_name == 'ACTNET':
        dataset = ActivityNetDataset(config, split_type, test_img_tfms)
        # ACTNET uses all_pairs list, extract unique video IDs and remove .mp4 suffix
        return list(dict.fromkeys([pair[0].replace('.mp4', '') for pair in dataset.all_pairs]))
            
    elif config.dataset_name == 'DIDEMO':
        dataset = DiDeMoDataset(config, split_type, test_img_tfms)
        # DiDeMo uses all_pairs list, extract unique video IDs and remove .mp4 suffix
        return list(dict.fromkeys([pair[0].replace('.mp4', '') for pair in dataset.all_pairs]))
    else:
        raise NotImplementedError(f"Dataset {config.dataset_name} not supported")


class VideoFeatureExtractor:
    """
    Extract and cache video frame-level CLIP embeddings.
    """

    def __init__(self, model, config, cache_dir: str, device='cuda'):
        """
        Args:
            model: XPool model (CLIPBaseline)
            config: Configuration object
            cache_dir: Directory to save cached features
            device: Device to run extraction on
        """
        self.model = model
        self.config = config
        self.cache_dir = cache_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Initialize image transforms
        img_transforms = init_transform_dict(config.input_res)
        self.img_transforms = img_transforms['clip_test']

        print(f"Feature extractor initialized")
        print(f"Device: {self.device}")
        print(f"Pooling type: {config.pooling_type}")
        print(f"Cache directory: {cache_dir}")

    def _get_video_path(self, video_id: str, videos_dir: str) -> str:
        """
        Construct dataset-specific video path.
        
        Args:
            video_id: Video identifier
            videos_dir: Base directory containing videos
            
        Returns:
            Full path to video file
        """
        if self.config.dataset_name == 'LSMDC':
            clip_prefix = video_id.split('.')[0][:-3]
            video_path = os.path.join(videos_dir, clip_prefix, video_id + '.avi')
        else:
            # Default structure for other datasets (MSRVTT, MSVD, ACTNET)
            video_path = os.path.join(videos_dir, video_id + '.mp4')
        
        return video_path

    def extract_video_features(self, video_id: str, video_path: str) -> np.ndarray:
        """
        Extract frame-level or video-level CLIP embeddings for a single video.
        
        If config.pooling_type == "avg", returns video-level features via average pooling.
        Otherwise, returns frame-level features.

        Args:
            video_id: Video identifier
            video_path: Path to video file

        Returns:
            Frame embeddings as numpy array [num_frames, embed_dim] or 
            Video embedding as numpy array [embed_dim] if pooling_type == "avg"
        """
        # Check if already cached
        cache_path = os.path.join(self.cache_dir, f"{video_id}.npz")
        if os.path.exists(cache_path):
            # Already cached, load and return
            cached_data = np.load(cache_path)
            if self.config.pooling_type == "avg":
                return cached_data['video_embed']
            else:
                return cached_data['frame_embeds']

        # Load video frames
        frames, frame_indices = VideoCapture.load_frames_from_video(
            video_path,
            self.config.num_frames,
            self.config.video_sample_type
        )

        # Apply transforms
        frames = self.img_transforms(frames)
        frames = frames.to(self.device)

        # Extract CLIP features
        with torch.no_grad():
            if self.config.huggingface:
                video_features = self.model.clip.get_image_features(frames)
            else:
                video_features = self.model.clip.encode_image(frames)

            # Normalize
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        frame_embeds = video_features.cpu().numpy()

        # Save to cache based on pooling type
        if self.config.pooling_type == "avg":
            # Average pool frame features to get video-level embedding
            video_embed = frame_embeds.mean(axis=0)
            np.savez_compressed(
                cache_path,
                video_embed=video_embed,
                frame_indices=frame_indices,
                video_id=video_id
            )
            return video_embed
        else:
            # Save frame-level features
            np.savez_compressed(
                cache_path,
                frame_embeds=frame_embeds,
                frame_indices=frame_indices,
                video_id=video_id
            )
            return frame_embeds

    def extract_from_video_list(self, video_ids: list, videos_dir: str):
        """
        Extract features for a list of video IDs.

        Args:
            video_ids: List of video IDs to process
            videos_dir: Directory containing video files
        """
        print(f"Extracting features for {len(video_ids)} videos...")

        for video_id in tqdm(video_ids, desc="Extracting features"):
            # Check if already cached
            cache_path = os.path.join(self.cache_dir, f"{video_id}.npz")
            if os.path.exists(cache_path):
                continue

            # Construct video path (dataset-specific)
            video_path = self._get_video_path(video_id, videos_dir)

            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue

            try:
                self.extract_video_features(video_id, video_path)
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                continue

        print(f"Feature extraction complete!")
        print(f"Features saved to: {self.cache_dir}")


def main():
    """
    Main function to run feature extraction.
    """
    # Parse arguments using AllConfig
    config = AllConfig()

    # Add custom arguments for feature extraction
    parser = argparse.ArgumentParser(description='Extract video features for caching')
    parser.add_argument('--cache_dir', type=str,
                        default='reranker/xpool/video_features_cache/CLIP4clip',
                        help='Directory to save cached features')
    parser.add_argument('--checkpoint', type=str,
                        default='reranker/xpool/clip4clip/lsmdc_model_best.pth',
                        help='Path to model checkpoint used for feature extraction')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process (for testing)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Dataset split to extract features from (train or test)')

    args, unknown = parser.parse_known_args()

    print("="*60)
    print("Video Feature Extraction Utility")
    print("="*60)
    print(f"Dataset: {config.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Pooling type: {config.pooling_type}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Video directory: {config.videos_dir}")
    print("="*60)

    # Create cache directory with dataset name
    cache_dir = os.path.join(args.cache_dir, config.dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = ModelFactory.get_model(config)

    # Load checkpoint if specified
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint not found at {args.checkpoint}")

    # Create feature extractor
    extractor = VideoFeatureExtractor(
        model=model,
        config=config,
        cache_dir=cache_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


    # Get unique video IDs directly from dataset (much faster than iterating data_loader)
    print(f"\nLoading {args.split} dataset...")
    unique_video_ids = get_unique_video_ids(config, args.split)
    print(f"Found {len(unique_video_ids)} unique videos")

    if args.max_videos is not None:
        unique_video_ids = unique_video_ids[:args.max_videos]
        print(f"Limiting to first {len(unique_video_ids)} videos")

    # Extract features directly from video files (matches evaluator.py exactly)
    extractor.extract_from_video_list(
        video_ids=unique_video_ids,
        videos_dir=config.videos_dir
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
