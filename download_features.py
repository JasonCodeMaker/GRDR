#!/usr/bin/env python3
"""
Download GRDR-TVR dataset components from Hugging Face Hub.

This script provides a convenient way to download specific components
of the GRDR-TVR dataset including InternVideo2 features, GRDR checkpoints,
Xpool reranker checkpoints, and Xpool video features.

Examples:
    # Download everything
    python download_features.py --all

    # Download only features for MSR-VTT and ActivityNet
    python download_features.py --features --datasets msrvtt actnet

    # Download GRDR checkpoints for all datasets
    python download_features.py --grdr

    # Download Xpool reranker checkpoints for specific dataset
    python download_features.py --xpool --datasets msrvtt
    
    # Download Xpool video features
    python download_features.py --xpool-features --datasets msrvtt

    # Download the full Panda release bundle
    python download_features.py --all --datasets panda
"""

import argparse
import shutil
import tempfile
from pathlib import Path


REPO_ID = "JasonCoderMaker/GRDR-TVR"
DATASETS = ["msrvtt", "actnet", "didemo", "panda"]
DATASET_LABELS = {
    "msrvtt": "MSRVTT",
    "actnet": "ACTNET",
    "didemo": "DIDEMO",
    "panda": "PANDA",
}
RELEASE_GRDR_CHECKPOINTS = {
    "msrvtt": "GRDR/msrvtt/best_model/best_model.pt",
    "actnet": "GRDR/actnet/best_model/best_model.pt",
    "didemo": "GRDR/didemo/best_model/best_model.pt",
    "panda": "GRDR/panda/best_model/best_model.pt",
}
XPOOL_FILES = {
    "actnet": "actnet_model_best.pth",
    "didemo": "didemo_model_best.pth",
    "msrvtt": "msrvtt9k_model_best.pth",
    "panda": "panda_2150k_s42_model_best.pth",
}


def canonical_xpool_cache_dir(output_dir, dataset):
    """Return the cache root expected by the release eval scripts."""
    root = Path(output_dir)
    if dataset == "panda":
        return root / "Xpool-Panda" / "PANDA"
    return root / "Xpool" / DATASET_LABELS[dataset]


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
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=f"InternVideo2/{dataset}/*",
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            print(f"✓ {dataset} features downloaded to {features_dir / dataset}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {dataset} InternVideo2 features") from e


def download_grdr_checkpoints(datasets, output_dir="./output/checkpoints"):
    """Download GRDR model checkpoints."""
    print(f"\n{'='*70}")
    print(" Downloading GRDR Checkpoints")
    print(f"{'='*70}\n")
    
    grdr_dir = Path(output_dir) / "GRDR"
    grdr_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        print(f"\n Downloading {dataset} GRDR checkpoint...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=f"GRDR/{dataset}/**",
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            print(f"✓ {dataset} GRDR checkpoint downloaded to {grdr_dir / dataset}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {dataset} GRDR checkpoint") from e


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
            from huggingface_hub import snapshot_download

            temp_dir = xpool_features_dir.parent
            label = DATASET_LABELS[dataset]
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=[f"Xpool_features/{dataset}/**", f"Xpool_features/{label}/**"],
                local_dir=temp_dir,
                local_dir_use_symlinks=False,
            )
            src_candidates = [
                temp_dir / "Xpool_features" / dataset,
                temp_dir / "Xpool_features" / label,
            ]
            src_path = next((path for path in src_candidates if path.exists()), None)
            if src_path is None:
                raise FileNotFoundError(
                    f"No Xpool_features payload found for {dataset}; expected one of "
                    f"{', '.join(str(path) for path in src_candidates)}"
                )
            dst_path = canonical_xpool_cache_dir(output_dir, dataset)
            if src_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.move(str(src_path), str(dst_path))
                if (temp_dir / "Xpool_features").exists():
                    shutil.rmtree(temp_dir / "Xpool_features", ignore_errors=True)
            print(f"✓ {dataset} Xpool features downloaded to {dst_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {dataset} Xpool features") from e


def download_xpool_checkpoints(datasets, output_dir="./reranker/xpool/ckpt"):
    """Download Xpool reranker checkpoints."""
    print(f"\n{'='*70}")
    print(" Downloading Xpool Reranker Checkpoints")
    print(f"{'='*70}\n")
    
    xpool_dir = Path(output_dir)
    xpool_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        if dataset not in XPOOL_FILES:
            print(f"⊘ Skipping {dataset} (no Xpool checkpoint)")
            continue
            
        filename = XPOOL_FILES[dataset]
        print(f"\n Downloading {dataset} Xpool checkpoint...")
        
        try:
            from huggingface_hub import hf_hub_download

            # Download to a temp location to strip the Xpool prefix
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=f"Xpool/{filename}",
                    local_dir=temp_dir,
                    local_dir_use_symlinks=False,
                )
                # Move from temp_dir/Xpool/{filename} to xpool_dir/{filename}
                dst_path = xpool_dir / filename
                shutil.copy2(file_path, dst_path)
            print(f"✓ {dataset} Xpool checkpoint downloaded to {dst_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {dataset} Xpool checkpoint") from e


def download_annotations(datasets, output_dir="."):
    """Download release annotations that are too large or optional for git."""
    selected = [dataset for dataset in datasets if dataset == "panda"]
    if not selected:
        return

    print(f"\n{'='*70}")
    print(" Downloading Dataset Annotations")
    print(f"{'='*70}\n")

    for dataset in selected:
        print(f"\n Downloading {dataset} annotations...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=f"data/{dataset}/**",
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            print(f"✓ {dataset} annotations downloaded to {Path(output_dir) / 'data' / dataset}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {dataset} annotations") from e


def _require_path(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def verify_release_artifacts(
    datasets,
    *,
    features_dir=None,
    grdr_dir=None,
    xpool_dir=None,
    xpool_features_dir=None,
    annotations_dir=None,
):
    """Fail fast if selected release artifacts are missing after download."""
    checks = []
    for dataset in datasets:
        if features_dir is not None:
            checks.append((Path(features_dir) / "InternVideo2" / dataset, f"{dataset} InternVideo2 features"))
        if grdr_dir is not None:
            ckpt = Path(grdr_dir) / RELEASE_GRDR_CHECKPOINTS[dataset]
            checks.append((ckpt, f"{dataset} GRDR checkpoint"))
            checks.append((Path(str(ckpt) + ".code"), f"{dataset} GRDR checkpoint sidecar .code"))
        if xpool_dir is not None:
            checks.append((Path(xpool_dir) / XPOOL_FILES[dataset], f"{dataset} X-Pool checkpoint"))
        if xpool_features_dir is not None:
            checks.append((canonical_xpool_cache_dir(xpool_features_dir, dataset), f"{dataset} X-Pool feature cache"))
        if annotations_dir is not None and dataset == "panda":
            checks.append(
                (
                    Path(annotations_dir) / "data/panda/video_retreival_caption",
                    "Panda annotations",
                )
            )

    for path, label in checks:
        _require_path(path, label)

    print("\nRelease artifact verification passed.")


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

  # Download Xpool checkpoints and features for DiDeMo
  python download_features.py --xpool --xpool-features --datasets didemo

  # Download all components for Panda
  python download_features.py --all --datasets panda

  # Download all components for DiDeMo
  python download_features.py --all --datasets didemo
        """
    )
    
    # Component selection
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all components (features, GRDR, Xpool checkpoints, Xpool features)",
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
        "--xpool-features",
        action="store_true",
        help="Download Xpool video features",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        help="Download release annotations that are not bundled in git (currently Panda)",
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
        default="./output/checkpoints",
        help="Output directory for GRDR checkpoints (default: ./output/checkpoints)",
    )
    parser.add_argument(
        "--xpool-dir",
        type=str,
        default="./reranker/xpool/ckpt",
        help="Output directory for Xpool checkpoints (default: ./reranker/xpool/ckpt)",
    )
    parser.add_argument(
        "--xpool-features-dir",
        type=str,
        default="./reranker/xpool/video_features_cache",
        help="Output directory for Xpool video features (default: ./reranker/xpool/video_features_cache)",
    )
    
    args = parser.parse_args()
    
    # Validate: at least one component must be selected
    if not any([args.all, args.features, args.grdr, args.xpool, args.xpool_features, args.annotations]):
        parser.error(
            "Please specify at least one component: --all, --features, --grdr, "
            "--xpool, --xpool-features, or --annotations"
        )
    
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
    
    if args.all or args.xpool_features:
        download_xpool_features(args.datasets, args.xpool_features_dir)

    if args.all or args.annotations:
        download_annotations(args.datasets)

    verify_release_artifacts(
        args.datasets,
        features_dir=args.features_dir if (args.all or args.features) else None,
        grdr_dir=args.grdr_dir if (args.all or args.grdr) else None,
        xpool_dir=args.xpool_dir if (args.all or args.xpool) else None,
        xpool_features_dir=args.xpool_features_dir if (args.all or args.xpool_features) else None,
        annotations_dir="." if (args.all or args.annotations) else None,
    )
    
    print(f"\n{'='*70}")
    print("✓ Download Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
