import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
from config.base_config import Config
from datasets.video_capture import VideoCapture


class PerQueryEvaluator:
    """
    Per-query evaluator for XPool model that processes one query at a time.

    This evaluator is designed for practical single-query inference while maintaining
    identical results to the batch evaluation pipeline in trainer.py::_valid_epoch_step.

    Key Features:
    - Process queries one at a time (on-the-fly query processing)
    - Two video feature modes: cached (pre-extracted) or on-the-fly extraction
    - On-the-fly pool_frames computation for each query
    - Detailed timing tracking for each processing step
    - Results aggregation matching original batch evaluation metrics

    Args:
        model: The XPool model (CLIPBaseline)
        config: Configuration object with model parameters
        video_ids: List of all video IDs in the test set (in order)
                   If candidates_file is provided, this will be overridden
        tokenizer: Text tokenizer (CLIP tokenizer or SimpleTokenizer)
        cache_dir: Directory containing cached video features (.npz files)
                   If None, features will be extracted on-the-fly from videos
        videos_dir: Directory containing video files (required for on-the-fly mode)
        candidates_file: Path to JSON file with per-query candidate lists
                         If provided, each query only searches its candidates
        device: Device to run computations on (default: 'cuda' if available)
        excluded_videos: List of video IDs to exclude from evaluation (e.g., problematic videos)
        video_batch_size: Number of videos to process at once during similarity computation
                          Reduces memory usage for large video pools (default: 1000)
    """

    def __init__(
        self,
        model,
        config: Config,
        video_ids: List[str],
        tokenizer,
        cache_dir: Optional[str] = None,
        videos_dir: Optional[str] = None,
        candidates_file: Optional[str] = None,
        device: Optional[str] = None,
        excluded_videos: Optional[List[str]] = None,
        video_batch_size: int = 1000
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.videos_dir = videos_dir or config.videos_dir
        self.candidates_file = candidates_file
        self.excluded_videos = set(excluded_videos) if excluded_videos else set()
        self.video_batch_size = video_batch_size

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Load candidate data if provided
        self.use_candidates = candidates_file is not None
        self.query_candidates_map = {}  # query_idx -> list of candidate video IDs
        self.query_text_to_idx = {}     # query_text -> query_idx for lookup

        if self.use_candidates:
            self._load_candidates_file(candidates_file)
            # Override video_ids with union of all candidates, excluding problematic videos
            all_candidates = set()
            for candidates in self.query_candidates_map.values():
                all_candidates.update(candidates)
            all_candidates -= self.excluded_videos
            self.video_ids = sorted(list(all_candidates))
            print(f"Loaded candidates file: {candidates_file}")
            print(f"  Total queries with candidates: {len(self.query_candidates_map)}")
            if self.excluded_videos:
                print(f"  Excluded videos: {len(self.excluded_videos)}")
            print(f"  Total unique candidate videos: {len(self.video_ids)}")
        else:
            # Filter excluded videos from full video list
            self.video_ids = [vid for vid in video_ids if vid not in self.excluded_videos]
            print(f"Full retrieval mode (no candidates filtering)")
            if self.excluded_videos:
                print(f"  Excluded videos: {len(self.excluded_videos)}")

        # Determine feature mode
        self.use_cache = cache_dir is not None
        if self.use_cache:
            print(f"Using cached video features from: {cache_dir}")
        else:
            print(f"Extracting video features on-the-fly from: {self.videos_dir}")

        # Storage for per-query results
        self.query_count = 0
        self.query_texts = []
        self.query_video_ids = []  # Ground truth video IDs for each query
        self.query_candidates = {}  # query_idx -> list of candidate video IDs used

        # Similarity storage: query_idx -> {video_id: similarity_score}
        self.query_similarities = {}

        # Timing statistics
        self.timing_stats = {
            'query_encode': [],
            'video_load': [],
            'frame_pooling': [],
            'similarity_compute': [],
            'total': []
        }

        print(f"Initialized evaluator with {len(self.video_ids)} videos in search pool")
        print(f"Video batch size: {self.video_batch_size} (for memory-efficient processing)")
        print("Note: Each query is independent - no feature caching between queries")

    def _load_candidates_file(self, candidates_file: str):
        """
        Load per-query candidate lists from JSON file.

        Expected format:
        {
            "results": [
                {
                    "query_text": "...",
                    "ground_truth_video_id": "...",
                    "candidates": ["video1", "video2", ...],
                    "num_candidates": 50
                },
                ...
            ]
        }
        """
        if not os.path.exists(candidates_file):
            raise FileNotFoundError(f"Candidates file not found: {candidates_file}")

        with open(candidates_file, 'r') as f:
            candidate_data = json.load(f)

        if 'results' not in candidate_data:
            raise ValueError(f"Candidates file missing 'results' key: {candidates_file}")

        results = candidate_data['results']

        for query_idx, result in enumerate(results):
            query_text = result['query_text']
            candidates = result['candidates']
            
            # Filter out excluded videos from candidates
            if self.excluded_videos:
                candidates = [vid for vid in candidates if vid not in self.excluded_videos]

            # Store mapping from query_idx to candidates
            self.query_candidates_map[query_idx] = candidates

            # Store mapping from query_text to query_idx for lookup
            self.query_text_to_idx[query_text] = query_idx

    def _normalize_video_id(self, video_id: str) -> str:
        """
        Normalize video ID by removing file extensions.
        
        Args:
            video_id: Video identifier that may include file extension
            
        Returns:
            Normalized video ID without file extension
        """
        if video_id.endswith('.mp4'):
            return video_id[:-4]
        if video_id.endswith('.avi'):
            return video_id[:-4]
        return video_id

    def _load_cached_features(self, video_id: str) -> Optional[torch.Tensor]:
        """
        Load cached video frame embeddings from disk.

        Args:
            video_id: Video identifier

        Returns:
            Frame embeddings tensor [num_frames, embed_dim] or None if not found
        """
        # Normalize video ID to match cache file naming
        normalized_id = self._normalize_video_id(video_id)
        cache_path = os.path.join(self.cache_dir, f"{normalized_id}.npz")

        if not os.path.exists(cache_path):
            return None

        try:
            cached_data = np.load(cache_path)
            frame_embeds = torch.from_numpy(cached_data['frame_embeds']).float()
            return frame_embeds
        except Exception as e:
            print(f"Error loading cached features for {video_id}: {e}")
            return None

    def _get_video_path(self, video_id: str) -> str:
        """
        Construct dataset-specific video path.
        
        Args:
            video_id: Video identifier (clip_id for LSMDC)
            
        Returns:
            Full path to video file
        """
        # Normalize video ID to ensure no double extensions
        normalized_id = self._normalize_video_id(video_id)
        
        # DiDeMo has special directory structure
        if self.config.dataset_name == 'DIDEMO':
            # DiDeMo test videos are in test/test_videos subdirectory
            video_path = os.path.join(self.videos_dir, 'test', 'test_videos', normalized_id + '.mp4')
        elif self.config.dataset_name == 'LSMDC':
            clip_prefix = normalized_id.split('.')[0][:-3]
            video_path = os.path.join(self.videos_dir, clip_prefix, normalized_id + '.avi')
        else:
            # Default structure for other datasets (MSRVTT, MSVD, ACTNET)
            video_path = os.path.join(self.videos_dir, normalized_id + '.mp4')
        
        return video_path

    def _extract_video_features(self, video_id: str) -> torch.Tensor:
        """
        Extract video frame embeddings on-the-fly from video file.

        This replicates the forward pass logic from CLIPBaseline for video encoding.

        Args:
            video_id: Video identifier

        Returns:
            Frame embeddings tensor [num_frames, embed_dim]
        """
        # Construct video path (dataset-specific)
        video_path = self._get_video_path(video_id)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load video frames using VideoCapture
        frames, _ = VideoCapture.load_frames_from_video(
            video_path,
            self.config.num_frames,
            self.config.video_sample_type
        )

        # Apply image transforms (same as dataset)
        from datasets.model_transforms import init_transform_dict
        img_transforms = init_transform_dict(self.config.input_res)
        test_img_tfms = img_transforms['clip_test']
        frames = test_img_tfms(frames)

        # Extract features using CLIP
        # frames shape: [num_frames, 3, H, W]
        frames = frames.to(self.device)

        with torch.no_grad():
            if self.config.huggingface:
                video_features = self.model.clip.get_image_features(frames)
            else:
                video_features = self.model.clip.encode_image(frames)

            # Normalize
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_features.cpu()  # [num_frames, embed_dim]

    def _get_video_features(self, video_id: str) -> torch.Tensor:
        """
        Get video frame features either from disk cache or by extracting on-the-fly.

        Note: No in-memory caching - each query loads features independently.

        Args:
            video_id: Video identifier

        Returns:
            Frame embeddings tensor [num_frames, embed_dim]
        """
        if self.use_cache:
            # Load from disk cache (.npz file)
            features = self._load_cached_features(video_id)
            if features is not None:
                return features
            else:
                # Fallback to on-the-fly extraction if cache miss
                print(f"Warning: Cache miss for {video_id}, extracting on-the-fly")
                return self._extract_video_features(video_id)
        else:
            # Extract from video file on-the-fly
            return self._extract_video_features(video_id)

    def evaluate_query(
        self,
        query_text: str,
        video_id_gt: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a single text query against all videos.

        This method replicates the per-query evaluation logic while maintaining
        compatibility with the original batch evaluation pipeline.

        Args:
            query_text: Text query string
            video_id_gt: Ground truth video ID (optional, for rank computation)

        Returns:
            Dictionary containing:
                - 'query_idx': Index of this query
                - 'query_text': The input query text
                - 'video_id_gt': Ground truth video ID (if provided)
                - 'similarities': Dict mapping video_id -> similarity score
                - 'ranked_videos': List of (video_id, score) sorted by score
                - 'rank': Rank of ground truth video (if video_id_gt provided)
                - 'timing': Dict with timing breakdown
                    - 'query_encode': Time to encode query
                    - 'video_load': Time to load/extract video features
                    - 'frame_pooling': Time for pool_frames operation
                    - 'similarity_compute': Time to compute similarities
                    - 'total': Total time for this query
        """
        total_start = time.time()
        timing = {}

        # Store query information
        query_idx = self.query_count
        self.query_texts.append(query_text)
        if video_id_gt is not None:
            self.query_video_ids.append(video_id_gt)

        # Step 1: Encode query text
        encode_start = time.time()
        with torch.no_grad():
            # Tokenize text
            if self.tokenizer is not None:
                text_data = self.tokenizer([query_text], return_tensors='pt',
                                          padding=True, truncation=True)
                if isinstance(text_data, torch.Tensor):
                    text_data = text_data.to(self.device)
                else:
                    text_data = {key: val.to(self.device) for key, val in text_data.items()}
            else:
                # Assume text is already tokenized
                text_data = query_text

            # Encode text
            if self.config.huggingface:
                text_embed = self.model.clip.get_text_features(**text_data)
            else:
                text_embed = self.model.clip.encode_text(text_data)

            # Normalize
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.cpu()  # [1, embed_dim]

        timing['query_encode'] = time.time() - encode_start

        # Step 2: Determine video search pool for this query
        # If candidates mode: use per-query candidates, else use all videos
        if self.use_candidates:
            # Look up candidate pool for this query
            if query_text in self.query_text_to_idx:
                candidate_query_idx = self.query_text_to_idx[query_text]
                search_pool = self.query_candidates_map[candidate_query_idx]
            else:
                # Query not in candidates file - use all videos as fallback
                search_pool = self.video_ids
                print(f"Warning: Query not found in candidates file, using full search pool")
        else:
            # Full retrieval mode
            search_pool = self.video_ids

        # Store which candidates were used for this query
        self.query_candidates[query_idx] = search_pool

        # Step 3: Process videos in batches to avoid OOM on large pools
        # This is memory-efficient: load batch -> pool -> compute sims -> free memory
        video_load_start = time.time()
        
        num_videos = len(search_pool)
        
        # Pre-allocate similarity array for all videos
        similarities_array = np.zeros(num_videos, dtype=np.float32)
        valid_video_ids = []
        
        # Process videos in batches
        pooling_time = 0.0
        sim_time = 0.0
        
        for batch_start in range(0, num_videos, self.video_batch_size):
            batch_end = min(batch_start + self.video_batch_size, num_videos)
            batch_pool = search_pool[batch_start:batch_end]
            
            # Load video features for this batch
            batch_video_embeds = []
            batch_valid_indices = []
            
            for local_idx, video_id in enumerate(batch_pool):
                try:
                    video_features = self._get_video_features(video_id)
                    batch_video_embeds.append(video_features)
                    batch_valid_indices.append(batch_start + local_idx)
                    valid_video_ids.append(video_id)
                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")
                    continue
            
            if len(batch_video_embeds) == 0:
                continue
            
            # Stack batch: [batch_size, num_frames, embed_dim]
            video_embeds_batch = torch.stack(batch_video_embeds, dim=0)
            
            # Move to device for pooling
            text_embed_device = text_embed.to(self.device)
            video_embeds_batch = video_embeds_batch.to(self.device)
            
            # Pool frames for this batch
            pooling_batch_start = time.time()
            with torch.no_grad():
                video_embeds_pooled = self.model.pool_frames(text_embed_device, video_embeds_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pooling_time += time.time() - pooling_batch_start
            
            # Compute similarities for this batch
            sim_batch_start = time.time()
            
            # Handle different pooling output shapes
            if video_embeds_pooled.dim() == 3:
                # topk/attention pooling: [batch_size, 1, embed_dim]
                video_embeds_pooled = video_embeds_pooled.squeeze(1)
            
            # Normalize
            text_embed_norm = text_embed_device / text_embed_device.norm(dim=-1, keepdim=True)
            video_embeds_pooled_norm = video_embeds_pooled / video_embeds_pooled.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarities: [1, batch_size]
            batch_similarities = torch.mm(text_embed_norm, video_embeds_pooled_norm.t()).squeeze(0)
            
            # Move to CPU and store in pre-allocated array
            batch_similarities_cpu = batch_similarities.cpu().numpy()
            for i, global_idx in enumerate(batch_valid_indices):
                if i < len(batch_similarities_cpu):
                    similarities_array[global_idx] = batch_similarities_cpu[i]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sim_time += time.time() - sim_batch_start
            
            # Free GPU memory immediately
            del video_embeds_batch, video_embeds_pooled, video_embeds_pooled_norm
            del batch_similarities, text_embed_device
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if len(valid_video_ids) == 0:
            raise ValueError("No valid video features found")
        
        timing['video_load'] = time.time() - video_load_start - pooling_time - sim_time
        timing['frame_pooling'] = pooling_time
        timing['similarity_compute'] = sim_time
        timing['total'] = time.time() - total_start

        # Store similarities for final metric computation
        # Only include valid videos (those that were successfully processed)
        similarity_dict = {}
        for vid_id, sim_score in zip(valid_video_ids, similarities_array[:len(valid_video_ids)]):
            similarity_dict[vid_id] = float(sim_score)

        self.query_similarities[query_idx] = similarity_dict

        # Rank videos by similarity
        ranked_videos = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

        # Compute rank of ground truth video
        rank = None
        if video_id_gt is not None:
            if video_id_gt in similarity_dict:
                # GT is in search pool - compute actual rank
                rank = next(i for i, (vid, _) in enumerate(ranked_videos) if vid == video_id_gt)
            else:
                # GT NOT in search pool (not in candidates) - set rank to 1000
                # This represents failure of first-stage retrieval
                rank = 1000

        # Update timing statistics
        for key, value in timing.items():
            self.timing_stats[key].append(value)

        self.query_count += 1

        return {
            'query_idx': query_idx,
            'query_text': query_text,
            'video_id_gt': video_id_gt,
            'similarities': similarity_dict,
            'ranked_videos': ranked_videos,
            'rank': rank,
            'timing': timing
        }

    def compute_final_metrics(self) -> Dict:
        """
        Compute final retrieval metrics across all evaluated queries.

        This method aggregates per-query results and computes metrics using
        the same logic as the original batch evaluation pipeline.

        For candidate mode: Computes metrics directly from per-query ranks
        For full retrieval: Uses similarity matrix approach

        Returns:
            Dictionary with metrics:
                - 'R1': Recall@1
                - 'R5': Recall@5
                - 'R10': Recall@10
                - 'R50': Recall@50
                - 'R100': Recall@100
                - 'MedR': Median rank
                - 'MeanR': Mean rank
                - 'timing_avg': Average timing statistics
        """
        if self.query_count == 0:
            raise ValueError("No queries have been evaluated yet")

        if self.use_candidates:
            # Candidate mode: Compute metrics directly from per-query ranks
            # This is more efficient and handles variable candidate pools
            metrics = self._compute_metrics_from_ranks()
        else:
            # Full retrieval mode: Use similarity matrix approach (original pipeline)
            metrics = self._compute_metrics_from_similarity_matrix()

        # Add timing statistics (mean only)
        timing_avg = {}
        for key, values in self.timing_stats.items():
            if values:
                timing_avg[key] = np.mean(values)

        metrics['timing_avg'] = timing_avg

        return metrics

    def _compute_metrics_from_ranks(self) -> Dict:
        """
        Compute metrics directly from per-query ranks.
        Used in candidate mode where each query has variable search pool.
        """
        # Collect all ranks
        ranks = []

        for query_idx in range(self.query_count):
            # Get rank for this query
            if query_idx < len(self.query_video_ids):
                video_id_gt = self.query_video_ids[query_idx]
                similarities = self.query_similarities[query_idx]

                if video_id_gt in similarities:
                    # GT in search pool - compute rank
                    ranked_videos = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                    rank = next(i for i, (vid, _) in enumerate(ranked_videos) if vid == video_id_gt)
                    ranks.append(rank)
                else:
                    # GT NOT in search pool - rank = 1000
                    ranks.append(1000)

        # Convert to numpy for metric computation
        ranks = np.array(ranks)

        # Compute metrics (same as modules.metrics.compute_metrics)
        metrics = {}
        metrics["R1"] = 100 * float(np.sum(ranks == 0)) / len(ranks)
        metrics["R5"] = 100 * float(np.sum(ranks < 5)) / len(ranks)
        metrics["R10"] = 100 * float(np.sum(ranks < 10)) / len(ranks)
        metrics["R50"] = 100 * float(np.sum(ranks < 50)) / len(ranks)
        metrics["R100"] = 100 * float(np.sum(ranks < 100)) / len(ranks)
        metrics["MedR"] = np.median(ranks) + 1
        metrics["MeanR"] = np.mean(ranks) + 1

        return metrics

    def _compute_metrics_from_similarity_matrix(self) -> Dict:
        """
        Compute metrics using similarity matrix approach.
        Used in full retrieval mode (original pipeline).
        """
        from modules.metrics import t2v_metrics

        # Create mapping from video_id to unique index
        unique_video_ids = list(set(self.video_ids))
        video_id_to_idx = {vid: idx for idx, vid in enumerate(unique_video_ids)}

        # Build per-video-id text groups (matching generate_embeds_per_video_id logic)
        # Group queries by their ground truth video ID
        all_vid_ids = []

        for query_idx in range(self.query_count):
            if query_idx < len(self.query_video_ids):
                vid_id = self.query_video_ids[query_idx]
                all_vid_ids.append(vid_id)

        # Construct similarity matrix: [num_unique_vids, max_queries_per_vid, num_unique_vids]
        # For simplicity in single-caption case, we can build: [num_queries, num_videos]

        num_queries = self.query_count
        num_videos = len(unique_video_ids)

        # Build similarity tensor
        sims_matrix = torch.zeros(num_queries, num_videos)

        for query_idx in range(num_queries):
            similarities = self.query_similarities[query_idx]
            for video_id, sim_score in similarities.items():
                if video_id in video_id_to_idx:
                    video_idx = video_id_to_idx[video_id]
                    sims_matrix[query_idx, video_idx] = sim_score

        # Handle per-video-id grouping (matching original pipeline)
        # Build text groups per video
        text_groups_per_video = {}
        for query_idx, vid_id in enumerate(all_vid_ids):
            if vid_id not in text_groups_per_video:
                text_groups_per_video[vid_id] = []
            text_groups_per_video[vid_id].append(query_idx)

        # Pad and create final similarity matrix matching t2v_metrics expected format
        max_text_per_vid = max(len(queries) for queries in text_groups_per_video.values())

        # Create similarity tensor: [num_vids, max_text_per_vid, num_vids]
        sims = torch.full((num_videos, max_text_per_vid, num_videos), float('-inf'))

        for vid_idx, vid_id in enumerate(unique_video_ids):
            if vid_id in text_groups_per_video:
                query_indices = text_groups_per_video[vid_id]
                for text_pos, query_idx in enumerate(query_indices):
                    # Copy similarities for this query
                    sims[vid_idx, text_pos, :] = sims_matrix[query_idx, :]

        # Compute metrics using original t2v_metrics function
        metrics = t2v_metrics(sims)

        return metrics

    def get_timing_summary(self) -> str:
        """
        Get a formatted string summary of timing statistics.

        Returns:
            Formatted timing summary string
        """
        if not self.timing_stats['total']:
            return "No timing data available"

        summary = "\n=== Timing Summary (mean per query) ===\n"
        for key, values in self.timing_stats.items():
            if values:
                mean_val = np.mean(values) * 1000  # Convert to milliseconds
                summary += f"{key:20s}: {mean_val:.2f}ms\n"

        return summary

    def reset(self):
        """Reset evaluator state for a new evaluation run."""
        self.query_count = 0
        self.query_texts = []
        self.query_video_ids = []
        self.query_similarities = {}
        for key in self.timing_stats:
            self.timing_stats[key] = []
