import os
import torch
import random
import numpy as np
import ujson as json
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture
from tqdm import tqdm


class CandidateDataset(Dataset):
    """
    Dataset for candidate-based reranking evaluation.
    Loads only candidate videos per query instead of the full corpus.
    
    Args:
        config: AllConfig object containing paths and parameters
        candidate_file: Path to JSON file with candidate lists per query
        split_type: 'train'/'test' (typically 'test' for reranking)
        img_transforms: Composition of image transforms
    """
    def __init__(self, config: Config, candidate_file, split_type='test', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        self.candidate_file = candidate_file
        
        # Set up video cache directory
        self.cache_dir = os.path.join("video_cache", config.dataset_name)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load candidate data
        with open(candidate_file, 'r') as f:
            candidate_data = json.load(f)
        
        self.candidate_results = candidate_data['results']
        
        # Build query-candidate pairs for iteration
        self.query_candidate_pairs = []
        self.query_texts = []
        self.unique_video_ids = set()
        
        for query_idx, result in enumerate(self.candidate_results):
            query_text = result['query_text']
            candidates = result['candidates']
            
            self.query_texts.append(query_text)
            
            # Store all unique video IDs to understand corpus size
            for candidate in candidates:
                self.unique_video_ids.add(candidate)
                # Store (query_idx, query_text, candidate_video_id) for __getitem__
                self.query_candidate_pairs.append((query_idx, query_text, candidate))
        
        print(f"Loaded {len(self.candidate_results)} queries with {len(self.query_candidate_pairs)} total query-candidate pairs")
        print(f"Unique videos in candidate set: {len(self.unique_video_ids)}")
        
        # Generate video cache for all unique videos
        self._generate_video_cache()

    def __len__(self):
        return len(self.query_candidate_pairs)

    def __getitem__(self, index):
        query_idx, query_text, video_id = self.query_candidate_pairs[index]
        
        # Try to load from cache first
        frames, indices = self._load_cached_video(video_id)
        
        if frames is None:
            # Fallback to direct loading if cache miss
            # Construct video path - handle both video7369 and video7369.mp4 formats
            if not video_id.endswith('.mp4'):
                video_path = os.path.join(self.videos_dir, video_id + '.mp4')
            else:
                video_path = os.path.join(self.videos_dir, video_id)
                video_id = video_id.replace('.mp4', '')  # Normalize for consistency

            # Load video frames
            frames, indices = VideoCapture.load_frames_from_video(
                video_path, 
                self.config.num_frames, 
                self.config.video_sample_type
            )

        # Apply transforms
        if self.img_transforms is not None:
            frames = self.img_transforms(frames)

        return {
            'video_id': video_id,
            'video': frames,
            'text': query_text,
            'query_idx': query_idx,  # Track which query this belongs to
        }

    def get_query_candidates(self, query_idx):
        """
        Get all candidate video IDs for a specific query.
        Used for batch processing queries with their candidates.
        """
        if query_idx < len(self.candidate_results):
            return self.candidate_results[query_idx]['candidates']
        return []

    def get_query_text(self, query_idx):
        """Get query text for a specific query index."""
        if query_idx < len(self.candidate_results):
            return self.candidate_results[query_idx]['query_text']
        return ""

    def get_num_queries(self):
        """Get total number of queries."""
        return len(self.candidate_results)
    
    def _get_cache_path(self, video_id):
        """Get cache file path for a video ID."""
        return os.path.join(self.cache_dir, f"{video_id}.npz")
    
    def _generate_video_cache(self):
        """Generate cache files for all unique videos that don't have cache."""
        cached_count = 0
        generated_count = 0
        
        print(f"Checking video cache for {len(self.unique_video_ids)} unique videos...")
        
        for video_id in tqdm(self.unique_video_ids, desc="Processing video cache"):
            cache_path = self._get_cache_path(video_id)
            
            if os.path.exists(cache_path):
                cached_count += 1
                continue
                
            # Cache miss - generate cache file
            if not video_id.endswith('.mp4'):
                video_path = os.path.join(self.videos_dir, video_id + '.mp4')
            else:
                video_path = os.path.join(self.videos_dir, video_id)
                
            if not os.path.exists(video_path):
                print(f"Warning: Video {video_path} not found, skipping cache generation")
                continue
                
            try:
                # Load video frames using original VideoCapture
                frames, frame_indices = VideoCapture.load_frames_from_video(
                    video_path, 
                    self.config.num_frames, 
                    self.config.video_sample_type
                )
                
                # Save to cache
                np.savez_compressed(cache_path, 
                                  frames=frames.numpy(), 
                                  indices=np.array(frame_indices))
                generated_count += 1
                
            except Exception as e:
                print(f"Error caching video {video_path}: {e}")
                continue
        
        print(f"Video cache status: {cached_count} already cached, {generated_count} newly generated")
    
    def _load_cached_video(self, video_id):
        """Load video frames from cache."""
        cache_path = self._get_cache_path(video_id)
        
        if not os.path.exists(cache_path):
            return None, None
            
        try:
            cached_data = np.load(cache_path)
            frames = torch.from_numpy(cached_data['frames'])
            indices = cached_data['indices'].tolist()
            return frames, indices
        except Exception as e:
            print(f"Error loading cached video {video_id}: {e}")
            return None, None

    def collate_by_query(self, query_idx):
        """
        Create a batch containing all candidates for a single query.
        This is the key method for handling variable candidate counts per query.
        
        Returns:
            Dictionary with batched data for one query and all its candidates
        """
        candidates = self.get_query_candidates(query_idx)
        query_text = self.get_query_text(query_idx)
        
        batch_videos = []
        batch_video_ids = []
        valid_candidates = []
        
        for candidate in candidates:
            # Handle video ID format
            if not candidate.endswith('.mp4'):
                video_id = candidate
            else:
                video_id = candidate.replace('.mp4', '')
            
            # Try to load from cache first
            frames, indices = self._load_cached_video(video_id)
            
            if frames is None:
                # Cache miss - fallback to direct loading
                if not candidate.endswith('.mp4'):
                    video_path = os.path.join(self.videos_dir, candidate + '.mp4')
                else:
                    video_path = os.path.join(self.videos_dir, candidate)
                
                # Check if video file exists
                if not os.path.exists(video_path):
                    print(f"Warning: Video {video_path} not found, skipping")
                    continue
                    
                try:
                    # Load video frames directly
                    frames, indices = VideoCapture.load_frames_from_video(
                        video_path, 
                        self.config.num_frames, 
                        self.config.video_sample_type
                    )
                    
                except Exception as e:
                    print(f"Error loading video {video_path}: {e}")
                    continue
            
            # Apply transforms
            if self.img_transforms is not None:
                frames = self.img_transforms(frames)
            
            batch_videos.append(frames)
            batch_video_ids.append(video_id)
            valid_candidates.append(candidate)
        
        if len(batch_videos) == 0:
            # Return empty batch if no valid videos
            return {
                'video_id': [],
                'video': torch.empty(0, self.config.num_frames, 3, 224, 224),
                'text': query_text,
                'query_idx': query_idx,
                'candidates': [],
                'num_candidates': 0
            }
        
        # Stack videos into batch tensor
        batch_videos = torch.stack(batch_videos, dim=0)
        
        return {
            'video_id': batch_video_ids,
            'video': batch_videos,  # Shape: [num_candidates, num_frames, 3, H, W]
            'text': query_text,
            'query_idx': query_idx,
            'candidates': valid_candidates,
            'num_candidates': len(valid_candidates)
        }


class CandidateDataLoader:
    """
    Custom data loader that yields one query at a time with all its candidates.
    This solves the variable batch size problem mentioned in Phase 4 of the plan.
    """
    def __init__(self, dataset: CandidateDataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_queries = dataset.get_num_queries()
        
    def __iter__(self):
        query_indices = list(range(self.num_queries))
        if self.shuffle:
            random.shuffle(query_indices)
            
        for query_idx in query_indices:
            batch = self.dataset.collate_by_query(query_idx)
            if batch['num_candidates'] > 0:  # Only yield if we have valid candidates
                yield batch
    
    def __len__(self):
        return self.num_queries