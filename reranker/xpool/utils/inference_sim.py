import os
import sys
import time
import json
import torch
import random
import argparse
import numpy as np
from typing import List
from dataclasses import dataclass, field
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class TimingResult:
    """Container for timing measurements of a single query."""
    query_encode: float      # Time for CLIP text encoding (seconds)
    video_load: float        # Time to load video features FROM DISK (seconds)
    frame_pooling: float     # Time for pool_frames operation (seconds)
    similarity_compute: float  # Time for cosine similarity computation (seconds)
    total: float             # Total end-to-end time (seconds)


@dataclass
class CorpusSizeResult:
    """Aggregated timing results for a specific corpus size."""
    corpus_size: int
    num_queries: int
    num_runs_per_query: int

    # Per-component statistics (mean and std in milliseconds)
    query_encode_mean: float
    query_encode_std: float
    video_load_mean: float
    video_load_std: float
    frame_pooling_mean: float
    frame_pooling_std: float
    similarity_compute_mean: float
    similarity_compute_std: float
    total_mean: float
    total_std: float


@dataclass
class SimulatorConfig:
    """Configuration for the latency simulator."""
    cache_dir: str = "reranker/xpool/video_features_cache/Xpool/MSRVTT"
    corpus_sizes: List[int] = field(default_factory=lambda: [1000, 5000, 10000, 20000, 50000, 100000])
    num_runs_per_query: int = 10        # Repeat each query N times for statistics
    num_warmup_queries: int = 5         # Warmup queries before timing
    num_test_queries: int = 100         # Number of queries to test per corpus size
    seed: int = 42                      # Random seed for reproducibility
    device: str = "cuda"                # Device for computation
    pooling_type: str = "avg"           # Frame pooling strategy
    video_batch_size: int = 5000        # Batch size for video processing (matches evaluator.py)
    embed_dim: int = 512                # CLIP embedding dimension
    num_frames: int = 12                # Frames per video
    feature_type: str = "frame"         # Feature type: "frame" or "video"


class InferenceLatencySimulator:
    """
    Simulates the XPool inference pipeline to measure per-query latency.

    This simulator replicates the EXACT inference path from PerQueryEvaluator:
    1. Text encoding (CLIP)
    2. Video feature loading FROM DISK for each query (no caching!)
    3. Frame pooling (model.pool_frames)
    4. Similarity computation (cosine similarity)

    CRITICAL: Each query loads ALL video features from disk independently,
    matching the actual evaluator behavior where there is NO feature caching
    between queries.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: SimulatorConfig,
    ):
        """
        Initialize the simulator.

        Args:
            model: XPool model (CLIPBaseline) with pool_frames method
            tokenizer: CLIP tokenizer for text encoding
            config: SimulatorConfig with simulation parameters
        """
        # Switch cache directory based on feature_type
        if config.feature_type == "video":
            # Replace Xpool with CLIP4clip while preserving dataset path
            # e.g., "...Xpool/MSRVTT" -> "...CLIP4clip/MSRVTT"
            config.cache_dir = config.cache_dir.replace("/Xpool/", "/CLIP4clip/")
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Move model to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        # Load available video IDs from cache directory
        self.available_video_ids = self._discover_cached_videos()
        self.num_available_videos = len(self.available_video_ids)

        print(f"Initialized InferenceLatencySimulator")
        print(f"  Device: {self.device}")
        print(f"  Cache directory: {config.cache_dir}")
        print(f"  Available cached videos: {self.num_available_videos}")
        print(f"  Feature type: {config.feature_type}")
        print(f"  Pooling type: {config.pooling_type}")
        print(f"  Video batch size: {config.video_batch_size}")
        print(f"  Runs per query: {config.num_runs_per_query}")
        print(f"  Warmup queries: {config.num_warmup_queries}")
        print(f"  IMPORTANT: Simulates real inference (disk I/O per query)")

    def _discover_cached_videos(self) -> List[str]:
        """
        Discover all cached video IDs from the cache directory.

        Returns:
            List of video IDs (without .npz extension)
        """
        video_ids = []
        cache_dir = self.config.cache_dir

        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

        for filename in os.listdir(cache_dir):
            if filename.endswith('.npz'):
                video_id = filename[:-4]  # Remove .npz extension
                video_ids.append(video_id)

        return sorted(video_ids)

    def _select_corpus_videos(self, corpus_size: int) -> List[str]:
        """
        Select video IDs for a given corpus size.

        For sizes <= num_available_videos: sample without replacement
        For sizes > num_available_videos: sample with replacement

        Args:
            corpus_size: Desired corpus size

        Returns:
            List of video IDs (may contain duplicates for large sizes)
        """
        if corpus_size <= self.num_available_videos:
            # Sample without replacement
            return random.sample(self.available_video_ids, corpus_size)
        else:
            # Sample with replacement (same video can appear multiple times)
            return random.choices(self.available_video_ids, k=corpus_size)

    def _load_single_video_features(self, video_id: str) -> torch.Tensor:
        """
        Load video features from a SINGLE .npz file.

        This is called for EACH video, for EACH query - matching the actual
        evaluator behavior where there's no caching between queries.

        Args:
            video_id: Video ID to load

        Returns:
            - If feature_type == "video": Tensor of shape [512,]
            - If feature_type == "frame": Tensor of shape [num_frames, embed_dim]
        """
        cache_path = os.path.join(self.config.cache_dir, f"{video_id}.npz")
        cached_data = np.load(cache_path)
        
        if self.config.feature_type == "video":
            # Video-level: load 'video_embed'
            video_embed = torch.from_numpy(cached_data['video_embed']).float()
            return video_embed
        else:
            # Frame-level: load 'frame_embeds'
            frame_embeds = torch.from_numpy(cached_data['frame_embeds']).float()
            return frame_embeds

    def _encode_text(self, query_text: str) -> torch.Tensor:
        """
        Encode text query using CLIP.

        Args:
            query_text: Text query string

        Returns:
            Normalized text embedding [1, embed_dim]
        """
        with torch.no_grad():
            # Tokenize
            text_data = self.tokenizer([query_text], return_tensors='pt',
                                        padding=True, truncation=True)
            text_data = {key: val.to(self.device) for key, val in text_data.items()}

            # Encode with CLIP
            text_embed = self.model.clip.get_text_features(**text_data)

            # Normalize
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        return text_embed

    def _run_single_query(
        self,
        query_text: str,
        video_ids: List[str],
    ) -> TimingResult:
        """
        Run a single query and measure timing for each component.

        IMPORTANT: This matches the EXACT behavior of PerQueryEvaluator.evaluate_query():
        - Loads video features from disk for EACH query (no caching)
        - Processes videos in batches to avoid OOM
        - Includes all disk I/O in timing

        Args:
            query_text: Text query to encode
            video_ids: List of video IDs to search (loaded from disk each time!)

        Returns:
            TimingResult with per-component timing
        """
        total_start = time.perf_counter()

        # Step 1: Text encoding
        encode_start = time.perf_counter()
        text_embed = self._encode_text(query_text)
        text_embed_cpu = text_embed.cpu()  # Keep CPU copy for reuse
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encode_time = time.perf_counter() - encode_start

        # Step 2 & 3 & 4: Process videos in batches (matching evaluator.py)
        # This is where the ACTUAL latency comes from - loading from disk!
        video_load_time = 0.0
        pooling_time = 0.0
        sim_time = 0.0

        num_videos = len(video_ids)
        batch_size = self.config.video_batch_size

        for batch_start in range(0, num_videos, batch_size):
            batch_end = min(batch_start + batch_size, num_videos)
            batch_video_ids = video_ids[batch_start:batch_end]

            # Step 2: Load video features from DISK for this batch
            load_start = time.perf_counter()
            batch_video_embeds = []
            for video_id in batch_video_ids:
                video_features = self._load_single_video_features(video_id)
                batch_video_embeds.append(video_features)

            # Stack batch based on feature type
            if self.config.feature_type == "video":
                # Stack video-level features: [batch_size, 512]
                video_embeds_batch = torch.stack(batch_video_embeds, dim=0)
            else:
                # Stack frame-level features: [batch_size, num_frames, embed_dim]
                video_embeds_batch = torch.stack(batch_video_embeds, dim=0)

            # Move to GPU
            video_embeds_batch = video_embeds_batch.to(self.device)
            text_embed_device = text_embed_cpu.to(self.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            video_load_time += time.perf_counter() - load_start

            # Step 3: Frame pooling (skip for video-level features)
            pooling_start = time.perf_counter()
            if self.config.feature_type == "video":
                # No pooling needed - already aggregated
                video_embeds_pooled = video_embeds_batch
            else:
                # Apply frame pooling
                with torch.no_grad():
                    video_embeds_pooled = self.model.pool_frames(text_embed_device, video_embeds_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pooling_time += time.perf_counter() - pooling_start

            # Step 4: Similarity computation
            sim_start = time.perf_counter()

            # Handle different pooling output shapes
            if video_embeds_pooled.dim() == 3:
                video_embeds_pooled = video_embeds_pooled.squeeze(1)

            # Normalize
            text_embed_norm = text_embed_device / text_embed_device.norm(dim=-1, keepdim=True)
            video_embeds_pooled_norm = video_embeds_pooled / video_embeds_pooled.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            batch_similarities = torch.mm(text_embed_norm, video_embeds_pooled_norm.t()).squeeze(0)

            # Move to CPU (matching evaluator behavior)
            _ = batch_similarities.cpu().numpy()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sim_time += time.perf_counter() - sim_start

            # Free GPU memory (matching evaluator behavior)
            del video_embeds_batch, video_embeds_pooled, video_embeds_pooled_norm
            del batch_similarities, text_embed_device
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.perf_counter() - total_start

        return TimingResult(
            query_encode=encode_time,
            video_load=video_load_time,
            frame_pooling=pooling_time,
            similarity_compute=sim_time,
            total=total_time
        )

    def _run_warmup(self, video_ids: List[str], sample_queries: List[str]):
        """
        Run warmup queries to handle cold start.

        This ensures CUDA kernels are compiled and OS file caches are populated
        before timing measurements begin.

        Args:
            video_ids: Video IDs to use for warmup
            sample_queries: Sample query texts for warmup
        """
        print(f"  Running {self.config.num_warmup_queries} warmup queries...")

        for i in range(self.config.num_warmup_queries):
            query = sample_queries[i % len(sample_queries)]
            _ = self._run_single_query(query, video_ids)

        # Synchronize GPU after warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_simulation_for_corpus_size(
        self,
        corpus_size: int,
        sample_queries: List[str],
    ) -> CorpusSizeResult:
        """
        Run simulation for a specific corpus size.

        Args:
            corpus_size: Number of videos in the corpus
            sample_queries: List of query texts to test

        Returns:
            CorpusSizeResult with aggregated timing statistics
        """
        print(f"\n{'='*60}")
        print(f"Testing corpus size: {corpus_size:,}")
        print(f"{'='*60}")

        # Select videos for this corpus size
        selected_videos = self._select_corpus_videos(corpus_size)
        print(f"  Selected {len(selected_videos)} videos (unique: {len(set(selected_videos))})")
        print(f"  Video batch size: {self.config.video_batch_size}")
        print(f"  NOTE: Features loaded from disk for EACH query (no caching)")

        # Run warmup (with smaller subset for faster warmup)
        warmup_videos = selected_videos[:min(1000, len(selected_videos))]
        self._run_warmup(warmup_videos, sample_queries)

        # Collect timing results
        all_timings: List[TimingResult] = []

        num_queries = min(self.config.num_test_queries, len(sample_queries))
        print(f"  Running {num_queries} queries x {self.config.num_runs_per_query} runs each...")

        for query_idx in tqdm(range(num_queries), desc="Queries"):
            query_text = sample_queries[query_idx]

            for _ in range(self.config.num_runs_per_query):
                timing = self._run_single_query(query_text, selected_videos)
                all_timings.append(timing)

        # Compute statistics
        result = self._compute_statistics(corpus_size, all_timings, num_queries)

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _compute_statistics(
        self,
        corpus_size: int,
        timings: List[TimingResult],
        num_queries: int,
    ) -> CorpusSizeResult:
        """
        Compute mean and std for each timing component.

        Args:
            corpus_size: Corpus size for this result
            timings: List of TimingResult from all runs
            num_queries: Number of queries tested

        Returns:
            CorpusSizeResult with aggregated statistics (in milliseconds)
        """
        # Convert to milliseconds
        to_ms = 1000.0

        query_encode = np.array([t.query_encode * to_ms for t in timings])
        video_load = np.array([t.video_load * to_ms for t in timings])
        frame_pooling = np.array([t.frame_pooling * to_ms for t in timings])
        similarity_compute = np.array([t.similarity_compute * to_ms for t in timings])
        total = np.array([t.total * to_ms for t in timings])

        return CorpusSizeResult(
            corpus_size=corpus_size,
            num_queries=num_queries,
            num_runs_per_query=self.config.num_runs_per_query,
            query_encode_mean=float(np.mean(query_encode)),
            query_encode_std=float(np.std(query_encode)),
            video_load_mean=float(np.mean(video_load)),
            video_load_std=float(np.std(video_load)),
            frame_pooling_mean=float(np.mean(frame_pooling)),
            frame_pooling_std=float(np.std(frame_pooling)),
            similarity_compute_mean=float(np.mean(similarity_compute)),
            similarity_compute_std=float(np.std(similarity_compute)),
            total_mean=float(np.mean(total)),
            total_std=float(np.std(total)),
        )

    def run_full_simulation(self, sample_queries: List[str]) -> List[CorpusSizeResult]:
        """
        Run simulation for all configured corpus sizes.

        Args:
            sample_queries: List of query texts to test

        Returns:
            List of CorpusSizeResult for each corpus size
        """
        results = []

        for corpus_size in self.config.corpus_sizes:
            result = self.run_simulation_for_corpus_size(corpus_size, sample_queries)
            results.append(result)

            # Print summary
            self._print_result_summary(result)

        return results

    def _print_result_summary(self, result: CorpusSizeResult):
        """Print formatted summary of timing results."""
        print(f"\n  Timing Summary (corpus size: {result.corpus_size:,})")
        print(f"  {'-'*50}")
        print(f"  {'Component':<25} {'Mean (ms)':<12} {'Std (ms)':<12}")
        print(f"  {'-'*50}")
        print(f"  {'Query Encode':<25} {result.query_encode_mean:>10.2f} {result.query_encode_std:>10.2f}")
        print(f"  {'Video Load (DISK I/O)':<25} {result.video_load_mean:>10.2f} {result.video_load_std:>10.2f}")
        print(f"  {'Frame Pooling':<25} {result.frame_pooling_mean:>10.2f} {result.frame_pooling_std:>10.2f}")
        print(f"  {'Similarity Compute':<25} {result.similarity_compute_mean:>10.2f} {result.similarity_compute_std:>10.2f}")
        print(f"  {'-'*50}")
        print(f"  {'TOTAL':<25} {result.total_mean:>10.2f} {result.total_std:>10.2f}")


def load_sample_queries(dataset_name: str = "MSRVTT", max_queries: int = 100) -> List[str]:
    """
    Load sample queries from the test dataset.

    Args:
        dataset_name: Dataset to load queries from
        max_queries: Maximum number of queries to load

    Returns:
        List of query text strings
    """
    import pandas as pd

    if dataset_name == "MSRVTT":
        test_csv = 'reranker/xpool/data/MSRVTT/MSRVTT_JSFUSION_test.csv'
        test_df = pd.read_csv(test_csv)
        queries = test_df['sentence'].tolist()[:max_queries]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

    return queries


def save_results_to_json(results: List[CorpusSizeResult], output_path: str):
    """
    Save simulation results to JSON file.

    Args:
        results: List of CorpusSizeResult objects
        output_path: Path to save JSON file
    """
    output_data = {
        "description": "Inference latency simulation results (includes disk I/O per query)",
        "note": "Video features loaded from disk for EACH query - no caching between queries",
        "results": []
    }

    for r in results:
        output_data["results"].append({
            "corpus_size": r.corpus_size,
            "num_queries": r.num_queries,
            "num_runs_per_query": r.num_runs_per_query,
            "timing_ms": {
                "query_encode": {"mean": r.query_encode_mean, "std": r.query_encode_std},
                "video_load_disk_io": {"mean": r.video_load_mean, "std": r.video_load_std},
                "frame_pooling": {"mean": r.frame_pooling_mean, "std": r.frame_pooling_std},
                "similarity_compute": {"mean": r.similarity_compute_mean, "std": r.similarity_compute_std},
                "total": {"mean": r.total_mean, "std": r.total_std},
            }
        })

    # Create parent directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Handle case where output_path is an existing directory
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, 'latency_simulation_results.json')

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_final_summary(results: List[CorpusSizeResult]):
    """
    Print final summary table comparing all corpus sizes.

    Args:
        results: List of CorpusSizeResult objects
    """
    print("\n" + "="*110)
    print("FINAL SUMMARY: Per-Query Latency by Corpus Size (all times in milliseconds)")
    print("NOTE: Video features loaded from disk for EACH query (no caching between queries)")
    print("="*110)
    print(f"{'Corpus':<12} {'Query Enc':<14} {'Video Load':<18} {'Pooling':<14} {'Similarity':<14} {'TOTAL':<16}")
    print(f"{'Size':<12} {'mean+/-std':<14} {'(DISK I/O)':<18} {'mean+/-std':<14} {'mean+/-std':<14} {'mean+/-std':<16}")
    print("-"*110)

    for r in results:
        print(f"{r.corpus_size:<12,} "
              f"{r.query_encode_mean:>5.1f}+/-{r.query_encode_std:<5.1f} "
              f"{r.video_load_mean:>7.1f}+/-{r.video_load_std:<7.1f} "
              f"{r.frame_pooling_mean:>5.1f}+/-{r.frame_pooling_std:<5.1f} "
              f"{r.similarity_compute_mean:>5.1f}+/-{r.similarity_compute_std:<5.1f} "
              f"{r.total_mean:>7.1f}+/-{r.total_std:<5.1f}")

    print("="*110)


def main():
    """Main entry point for the latency simulator."""
    parser = argparse.ArgumentParser(description='Inference Latency Simulator for XPool')
    parser.add_argument('--cache_dir', type=str,
                        default='reranker/xpool/video_features_cache/CLIP4clip/MSRVTT',
                        help='Directory with cached video features')
    parser.add_argument('--corpus_sizes', type=str, default='4917, 9384, 10000, 14926, 102055',
                        help='Comma-separated list of corpus sizes to test')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs per query for statistical robustness')
    parser.add_argument('--num_warmup', type=int, default=5,
                        help='Number of warmup queries before timing')
    parser.add_argument('--num_queries', type=int, default=20,
                        help='Number of queries to test per corpus size')
    parser.add_argument('--pooling_type', type=str, default='attention',
                        choices=['avg', 'topk', 'attention'],
                        help='Frame pooling strategy')
    parser.add_argument('--video_batch_size', type=int, default=5000,
                        help='Batch size for video processing (matches evaluator.py default)')
    parser.add_argument('--output', type=str, default='output/reranker/latency_simulation_results',
                        help='Output path for results JSON')
    parser.add_argument('--checkpoint', type=str,
                        default='reranker/xpool/ckpt/msrvtt9k_model_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--feature_type', type=str, default='video',
                        choices=['frame', 'video'],
                        help='Feature type: "frame" for Xpool (frame-level), "video" for CLIP4clip (video-level)')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Parse corpus sizes
    corpus_sizes = [int(x.strip()) for x in args.corpus_sizes.split(',')]

    # Create config
    sim_config = SimulatorConfig(
        cache_dir=args.cache_dir,
        corpus_sizes=corpus_sizes,
        num_runs_per_query=args.num_runs,
        num_warmup_queries=args.num_warmup,
        num_test_queries=args.num_queries,
        seed=args.seed,
        pooling_type=args.pooling_type,
        video_batch_size=args.video_batch_size,
        feature_type=args.feature_type,
    )

    print("="*60)
    print("Inference Latency Simulator")
    print("="*60)
    print(f"Feature type: {sim_config.feature_type}")
    print(f"Cache directory: {sim_config.cache_dir}")
    print(f"Corpus sizes: {corpus_sizes}")
    print(f"Runs per query: {sim_config.num_runs_per_query}")
    print(f"Warmup queries: {sim_config.num_warmup_queries}")
    print(f"Test queries: {sim_config.num_test_queries}")
    print(f"Pooling type: {sim_config.pooling_type}")
    print(f"Video batch size: {sim_config.video_batch_size}")
    print("-"*60)
    print("IMPORTANT: This simulates REAL inference behavior!")
    print("Video features are loaded from disk for EACH query.")
    print("="*60)

    # Load tokenizer
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Load model
    from config.all_config import AllConfig
    from model.model_factory import ModelFactory

    # Create config for model loading with required parameters
    model_config = AllConfig()
    model_config.pooling_type = args.pooling_type

    print("\nLoading model...")
    model = ModelFactory.get_model(model_config)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found: {args.checkpoint}")
        print("Using randomly initialized model")

    # Load sample queries
    print("\nLoading sample queries...")
    sample_queries = load_sample_queries("MSRVTT", max_queries=args.num_queries)
    print(f"Loaded {len(sample_queries)} queries")

    # Create simulator
    simulator = InferenceLatencySimulator(
        model=model,
        tokenizer=tokenizer,
        config=sim_config,
    )

    # Run simulation
    results = simulator.run_full_simulation(sample_queries)

    # Print final summary
    print_final_summary(results)

    # Save results
    save_results_to_json(results, args.output)

    print("\nSimulation complete!")


if __name__ == '__main__':
    main()
