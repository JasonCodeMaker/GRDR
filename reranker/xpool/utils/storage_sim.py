"""
Storage Simulator for Feature Representations

Compares storage requirements of:
1. SemanticID: [4, 3] int16 array
2. Video Level Feature: [512] float32 array
3. Frame Level Feature: [12, 512] float32 array
"""

import numpy as np
import os
import tempfile
from typing import Dict, Tuple

# Feature specifications
FEATURE_SPECS = {
    "SemanticID": {"shape": (1, 4), "dtype": np.int16},
    "VideoLevelFeature": {"shape": (512,), "dtype": np.float32},
    "FrameLevelFeature": {"shape": (12, 512), "dtype": np.float32},
}

# Corpus sizes to simulate
# CORPUS_SIZES = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
CORPUS_SIZES = [1_000,10_000]

def generate_features(feature_type: str, num_samples: int) -> np.ndarray:
    """Generate random features based on feature type specification."""
    spec = FEATURE_SPECS[feature_type]
    shape = (num_samples,) + spec["shape"]
    dtype = spec["dtype"]

    if dtype == np.int16:
        # SemanticID: simulate codebook indices (0-127 for typical codebook)
        return np.random.randint(0, 128, size=shape, dtype=dtype)
    else:
        # Float features: normalized random features
        return np.random.randn(*shape).astype(dtype)


def measure_storage(features: np.ndarray, save_path: str, compressed: bool = False) -> float:
    """Save features as npz and return file size in KB."""
    if compressed:
        np.savez_compressed(save_path, features=features)
    else:
        np.savez(save_path, features=features)

    file_size_bytes = os.path.getsize(save_path + ".npz")
    return file_size_bytes / 1024  # Convert to KB


def format_size(size_kb: float) -> str:
    """Format size in MB (consistent unit for all features)."""
    size_mb = size_kb / 1024
    return f"{size_mb:.2f} MB"


def calculate_theoretical_storage(feature_type: str, num_samples: int) -> float:
    """Calculate theoretical storage in KB (without npz overhead)."""
    spec = FEATURE_SPECS[feature_type]
    bytes_per_element = np.dtype(spec["dtype"]).itemsize
    elements_per_sample = np.prod(spec["shape"])
    total_bytes = num_samples * elements_per_sample * bytes_per_element
    return total_bytes / 1024  # KB


def simulate_storage(corpus_sizes: list = None, sample_size: int = 1000) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """
    Run storage simulation for all feature types and corpus sizes.

    Uses a sample-based approach: generate sample_size samples, measure storage,
    then extrapolate to full corpus sizes.

    Returns:
        Dict mapping feature_type -> corpus_size -> (actual_kb, theoretical_kb)
    """
    if corpus_sizes is None:
        corpus_sizes = CORPUS_SIZES

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for feature_type in FEATURE_SPECS.keys():
            results[feature_type] = {}

            # Generate sample features
            sample_features = generate_features(feature_type, sample_size)

            # Measure actual storage for sample
            save_path = os.path.join(tmpdir, f"{feature_type}_sample")
            sample_kb = measure_storage(sample_features, save_path, compressed=False)

            # Per-sample storage (with npz overhead amortized)
            kb_per_sample = sample_kb / sample_size

            # Theoretical per-sample storage
            theoretical_per_sample = calculate_theoretical_storage(feature_type, 1)

            for corpus_size in corpus_sizes:
                actual_kb = kb_per_sample * corpus_size
                theoretical_kb = theoretical_per_sample * corpus_size
                results[feature_type][corpus_size] = (actual_kb, theoretical_kb)

    return results


def print_results(results: Dict[str, Dict[int, Tuple[float, float]]]) -> None:
    """Print simulation results in a formatted table."""
    corpus_sizes = sorted(list(results["SemanticID"].keys()))

    print("=" * 100)
    print("                              Storage Simulation Results")
    print("=" * 100)
    print()

    # Header
    print(f"{'Corpus Size':<15} {'SemanticID':<20} {'Video Feature':<20} {'Frame Feature':<20}")
    print("-" * 100)

    # Data rows
    for size in corpus_sizes:
        sid_kb = results["SemanticID"][size][0]
        vid_kb = results["VideoLevelFeature"][size][0]
        frame_kb = results["FrameLevelFeature"][size][0]

        size_str = f"{size:,}"
        print(f"{size_str:<15} {format_size(sid_kb):<20} {format_size(vid_kb):<20} {format_size(frame_kb):<20}")

    print("=" * 100)
    print()

    # Storage ratios
    print("Storage Compression Ratio (compared to SemanticID):")
    print("-" * 50)

    # Use 1M corpus for ratio calculation
    ref_size = max(corpus_sizes)
    sid_kb = results["SemanticID"][ref_size][0]
    vid_kb = results["VideoLevelFeature"][ref_size][0]
    frame_kb = results["FrameLevelFeature"][ref_size][0]

    print(f"  Video Level Feature: {vid_kb / sid_kb:.1f}x more storage")
    print(f"  Frame Level Feature: {frame_kb / sid_kb:.1f}x more storage")
    print()

    # Per-video storage breakdown
    print("Per-Video Storage (bytes):")
    print("-" * 50)
    for feature_type, spec in FEATURE_SPECS.items():
        bytes_per_element = np.dtype(spec["dtype"]).itemsize
        elements = np.prod(spec["shape"])
        total_bytes = elements * bytes_per_element
        print(f"  {feature_type:<20}: {int(total_bytes):>6} bytes  (shape: {spec['shape']}, dtype: {spec['dtype'].__name__})")

    print("=" * 100)


def save_sample_features(output_dir: str, num_samples: int = 100) -> Dict[str, str]:
    """
    Generate and save sample features for each type.

    Returns:
        Dict mapping feature_type -> saved file path
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    for feature_type in FEATURE_SPECS.keys():
        features = generate_features(feature_type, num_samples)
        save_path = os.path.join(output_dir, f"{feature_type}_{num_samples}_samples")
        np.savez(save_path, features=features)
        saved_files[feature_type] = save_path + ".npz"

        file_size_kb = os.path.getsize(saved_files[feature_type]) / 1024
        print(f"Saved {feature_type}: {saved_files[feature_type]} ({format_size(file_size_kb)})")

    return saved_files


def main():
    """Main entry point for storage simulation."""
    print("\nGenerating sample features and measuring storage...\n")

    # Run simulation
    results = simulate_storage(CORPUS_SIZES)

    # Print results
    print_results(results)

    # Optionally save sample files for verification
    print("\nSaving sample files for verification...")
    output_dir = os.path.join(os.path.dirname(__file__), "storage_samples")
    save_sample_features(output_dir, num_samples=100)
    print(f"\nSample files saved to: {output_dir}")


if __name__ == "__main__":
    main()
