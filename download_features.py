#!/usr/bin/env python3
"""
Download GRDR-TVR dataset components from Hugging Face Hub.

This script provides a convenient way to download specific components
of the GRDR-TVR dataset including InternVideo2 features, GRDR checkpoints,
and Xpool reranker models.

Examples:
    # Download everything
    python download_features.py --all

    # Download only features for MSR-VTT and ActivityNet
    python download_features.py --features --datasets msrvtt actnet

    # Download GRDR checkpoints for all datasets
    python download_features.py --grdr

    # Download Xpool reranker for specific dataset
    python download_features.py --xpool --datasets msrvtt
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm


REPO_ID = "JasonCoderMaker/GRDR-TVR"
DATASETS = ["msrvtt", "actnet", "didemo", "lsmdc"]


def download_internvideo2_features(datasets, output_dir="./dataset/features"):
    """Download InternVideo2 pre-extracted features."""
    print(f"\n{'='*70}")
    print(" Downloading InternVideo2 Features")
    print(f"{'='*70}\n")
    
    features_dir = Path(output_dir) / "InternVideo2"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        print(f"\n Downloading {dataset} features...")
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=f"InternVideo2/{dataset}/*",
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            print(f"✓ {dataset} features downloaded to {features_dir / dataset}")
        except Exception as e:
            print(f"✗ Error downloading {dataset} features: {e}")


def download_grdr_checkpoints(datasets, output_dir="./output"):
    """Download GRDR model checkpoints."""
    print(f"\n{'='*70}")
    print(" Downloading GRDR Checkpoints")
    print(f"{'='*70}\n")
    
    grdr_dir = Path(output_dir) / "GRDR"
    grdr_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        print(f"\n Downloading {dataset} GRDR checkpoint...")
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=f"GRDR/{dataset}/**",
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            print(f"✓ {dataset} GRDR checkpoint downloaded to {grdr_dir / dataset}")
        except Exception as e:
            print(f"✗ Error downloading {dataset} GRDR checkpoint: {e}")


def download_xpool_features(datasets, output_dir="./reranker/xpool/video_features_cache"):
    """Download Xpool video features."""
    print(f"\n{'='*70}")
    print(" Downloading Xpool Video Features")
    print(f"{'='*70}\n")
    
    xpool_features_dir = Path(output_dir)
    xpool_features_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        print(f"\n Downloading {dataset} Xpool features...")
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=f"Xpool_features/{dataset}/**",
                local_dir=xpool_features_dir.parent,
                local_dir_use_symlinks=False,
            )
            print(f"✓ {dataset} Xpool features downloaded to {xpool_features_dir / dataset}")
        except Exception as e:
            print(f"✗ Error downloading {dataset} Xpool features: {e}")


def download_xpool_checkpoints(datasets, output_dir="./reranker/xpool/ckpt"):
    """Download Xpool reranker checkpoints."""
    print(f"\n{'='*70}")
    print(" Downloading Xpool Reranker Checkpoints")
    print(f"{'='*70}\n")
    
    xpool_dir = Path(output_dir)
    xpool_dir.mkdir(parents=True, exist_ok=True)
    
    xpool_files = {
        "actnet": "actnet_model_best.pth",
        "didemo": "didemo_model_best.pth",
        "lsmdc": "lsmdc_model_best.pth",
        "msrvtt": "msrvtt9k_model_best.pth",
    }
    
    for dataset in datasets:
        if dataset not in xpool_files:
            print(f"⊘ Skipping {dataset} (no Xpool checkpoint)")
            continue
            
        filename = xpool_files[dataset]
        print(f"\n Downloading {dataset} Xpool checkpoint...")
        
        try:
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"Xpool/{filename}",
                local_dir=xpool_dir.parent.parent,
                local_dir_use_symlinks=False,
            )
            print(f"✓ {dataset} Xpool checkpoint downloaded to {xpool_dir / filename}")
        except Exception as e:
            print(f"✗ Error downloading {dataset} Xpool checkpoint: {e}")


def download_scripts(output_dir="./scripts"):
    """Download utility scripts."""
    print(f"\n{'='*70}")
    print(" Downloading Utility Scripts")
    print(f"{'='*70}\n")
    
    scripts_dir = Path(output_dir)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    script_files = [
        "download_features.py",
        "download_checkpoints.sh",
    ]
    
    for script in script_files:
        try:
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=script,
                local_dir=".",
                local_dir_use_symlinks=False,
            )
            print(f"✓ {script} downloaded")
        except Exception as e:
            print(f"⊘ {script} not available: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download GRDR-TVR dataset components from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything for all datasets
  python download_features.py --all

  # Download only InternVideo2 features for MSR-VTT
  python download_features.py --features --datasets msrvtt

  # Download GRDR checkpoints for MSR-VTT and ActivityNet
  python download_features.py --grdr --datasets msrvtt actnet

  # Download all components for DiDeMo
  python download_features.py --all --datasets didemo
        """
    )
    
    # Component selection
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all components (features, GRDR, Xpool)",
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="Download InternVideo2 features",
    )
    parser.add_argument(
        "--grdr",
        action="store_true",
        help="Download GRDR model checkpoints",
    )
    parser.add_argument(
        "--xpool",
        action="store_true",
        help="Download Xpool reranker checkpoints",
    )
    parser.add_argument(
        "--scripts",
        action="store_true",
        help="Download utility scripts",
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to download (default: all)",
    )
    
    # Output directories
    parser.add_argument(
        "--features-dir",
        type=str,
        default="./dataset/features",
        help="Output directory for features (default: ./dataset/features)",
    )
    parser.add_argument(
        "--grdr-dir",
        type=str,
        default="./output",
        help="Output directory for GRDR checkpoints (default: ./output)",
    )
    parser.add_argument(
        "--xpool-dir",
        type=str,
        default="./reranker/xpool/ckpt",
        help="Output directory for Xpool checkpoints (default: ./reranker/xpool/ckpt)",
    )
    
    args = parser.parse_args()
    
    # Validate: at least one component must be selected
    if not any([args.all, args.features, args.grdr, args.xpool, args.scripts]):
        parser.error("Please specify at least one component: --all, --features, --grdr, --xpool, or --scripts")
    
    print(f"\n{'='*70}")
    print(f"GRDR-TVR Dataset Downloader")
    print(f"{'='*70}")
    print(f"Repository: {REPO_ID}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"{'='*70}\n")
    
    # Download components
    if args.all or args.features:
        download_internvideo2_features(args.datasets, args.features_dir)
    
    if args.all or args.grdr:
        download_grdr_checkpoints(args.datasets, args.grdr_dir)
    
    if args.all or args.xpool:
        download_xpool_checkpoints(args.datasets, args.xpool_dir)
    
    if args.scripts:
        download_scripts()
    
    print(f"\n{'='*70}")
    print("✓ Download Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
