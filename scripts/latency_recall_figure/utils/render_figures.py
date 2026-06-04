#!/usr/bin/env python
"""Render Panel A (CanHit@100 vs stage1_latency_ms) + Panel B (R@K vs total_latency_ms) per dataset.

Reads figure_data.csv (from make_figure.sh aggregate) and writes one PNG per dataset
to output/evaluation_results/figures/figures/. One subplot pair per dataset.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = ["grdr_ref", "tiger", "avg", "t2vindexer", "eercf", "hnsw", "ivf"]
METHOD_LABEL = {
    "grdr_ref": "GRDR",
    "tiger": "TIGER",
    "avg": "AVG",
    "t2vindexer": "T2VIndexer",
    "eercf": "EERCF",
    "hnsw": "HNSW",
    "ivf": "IVF",
}
METHOD_COLOR = {
    "grdr_ref": "#d62728",
    "tiger": "#1f77b4",
    "avg": "#2ca02c",
    "t2vindexer": "#9467bd",
    "eercf": "#ff7f0e",
    "hnsw": "#8c564b",
    "ivf": "#e377c2",
}
DATASET_ORDER = ["MSRVTT", "ACTNET", "DIDEMO", "LSMDC"]
R_AT_K = [1, 5, 10]


def numeric_or_nan(s):
    return pd.to_numeric(s, errors="coerce")


def render_one(df_ds: pd.DataFrame, ds: str, out_path: Path) -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{ds} — Setting 2 (in-distribution, C=128/L=3)", fontsize=14)

    # Panel A: CanHit@100 vs stage1_latency_ms
    for method in METHOD_ORDER:
        sub = df_ds[df_ds["method"] == method].copy()
        sub["stage1_latency_ms"] = numeric_or_nan(sub["stage1_latency_ms"])
        sub["CanHit@100"] = numeric_or_nan(sub["CanHit@100"])
        sub = sub.dropna(subset=["stage1_latency_ms", "CanHit@100"])
        sub = sub.sort_values("stage1_latency_ms")
        if sub.empty:
            continue
        axA.plot(
            sub["stage1_latency_ms"], sub["CanHit@100"],
            marker="o", linewidth=2, label=METHOD_LABEL[method],
            color=METHOD_COLOR[method],
        )
    axA.set_xscale("log")
    axA.set_xlabel("Stage-1 latency (ms, log)")
    axA.set_ylabel("CanHit@100 (%)")
    axA.set_title("Panel A — Stage-1 effectiveness vs efficiency")
    axA.grid(True, which="both", alpha=0.3)
    axA.legend(loc="best", fontsize=9)

    # Panel B: R@K vs total_latency_ms, K=1/5/10. Use marker shape for K, color for method.
    K_MARKER = {1: "o", 5: "s", 10: "^"}
    for method in METHOD_ORDER:
        sub = df_ds[df_ds["method"] == method].copy()
        sub["total_latency_ms"] = numeric_or_nan(sub["total_latency_ms"])
        for K in R_AT_K:
            col = f"R@{K}"
            if col not in sub:
                continue
            s2 = sub.copy()
            s2[col] = numeric_or_nan(s2[col])
            s2 = s2.dropna(subset=["total_latency_ms", col]).sort_values("total_latency_ms")
            if s2.empty:
                continue
            axB.plot(
                s2["total_latency_ms"], s2[col],
                marker=K_MARKER[K], linewidth=1.5,
                label=f"{METHOD_LABEL[method]} R@{K}",
                color=METHOD_COLOR[method],
                alpha=0.6 if K != 10 else 1.0,
            )
    axB.set_xscale("log")
    axB.set_xlabel("Total latency (ms, log)")
    axB.set_ylabel("Recall@K (%)")
    axB.set_title("Panel B — End-to-end effectiveness vs efficiency")
    axB.grid(True, which="both", alpha=0.3)
    axB.legend(loc="best", fontsize=7, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            print(f"skip {ds}: no rows", flush=True)
            continue
        render_one(sub, ds, args.out_dir / f"{ds.lower()}_panel_AB.png")


if __name__ == "__main__":
    main()
