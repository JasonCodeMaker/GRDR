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

METHOD_ORDER = ["grdr_ref", "tiger", "avg", "t2vindexer", "eercf", "hnsw", "ivf", "ivfpq", "opq"]
METHOD_LABEL = {
    "grdr_ref": "GRDR",
    "tiger": "TIGER",
    "avg": "AVG",
    "t2vindexer": "T2VIndexer",
    "eercf": "EERCF",
    "hnsw": "HNSW",
    "ivf": "IVF",
    "ivfpq": "IVF-PQ 16m",
    "opq": "OPQ 16m",
}
METHOD_COLOR = {
    "grdr_ref": "#d62728",
    "tiger": "#1f77b4",
    "avg": "#2ca02c",
    "t2vindexer": "#9467bd",
    "eercf": "#ff7f0e",
    "hnsw": "#8c564b",
    "ivf": "#e377c2",
    "ivfpq": "#7f7f7f",
    "opq": "#bcbd22",
}
DATASET_ORDER = ["MSRVTT", "ACTNET", "DIDEMO", "PANDA"]
R_AT_K = [1, 5, 10]
GR_BASELINE_METHODS = {"tiger", "avg", "t2vindexer"}
# Datasets whose GR baselines should be collapsed to a single best point.
# Keep this empty for the paper figure so Panda TIGER/AVG show the full op curve.
SELECT_BEST_GR_DATASETS = set()
PANDA_SINGLE_POINT_METHODS = {"hnsw", "ivf"}
MIDPOINT_METHODS = {"ivfpq", "opq"}
MIDPOINT_OP_VALUE = 100
AXIS_X_MARGIN = 0.14
AXIS_Y_MARGIN = 0.10


def numeric_or_nan(s):
    return pd.to_numeric(s, errors="coerce")


def apply_axis_margins(ax, x: float = AXIS_X_MARGIN, y: float = AXIS_Y_MARGIN) -> None:
    """Add visual padding so markers do not sit on plot borders."""
    ax.margins(x=x, y=y)
    ax.autoscale_view()


def select_best_gr_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Optionally collapse selected GR-baseline datasets to their best beam point."""
    parts = []
    selected = []
    for (ds, setting, method), group in df.groupby(["dataset", "setting", "method"], sort=False):
        if ds not in SELECT_BEST_GR_DATASETS or method not in GR_BASELINE_METHODS:
            parts.append(group)
            continue
        valid = group.copy()
        for col in ("R@1", "R@5", "R@10", "total_latency_ms", "op_point_value"):
            valid[col] = numeric_or_nan(valid[col])
        valid = valid.dropna(subset=["R@1"])
        if valid.empty:
            parts.append(group)
            continue
        best = valid.sort_values(
            ["R@1", "R@5", "R@10", "total_latency_ms", "op_point_value"],
            ascending=[False, False, False, True, True],
        ).head(1)
        parts.append(best)
        row = best.iloc[0]
        selected.append(
            f"{method}/{ds}/s{setting}: beam={int(row['op_point_value'])}, "
            f"cap={int(row['op_point_value']) * 3}, R@1={row['R@1']:.2f}"
        )
    if selected:
        print("selected Panda GR baseline points: " + "; ".join(selected), flush=True)
    return pd.concat(parts, ignore_index=True) if parts else df


def select_display_points(df: pd.DataFrame) -> pd.DataFrame:
    """Apply figure-only point selection without changing the underlying CSV."""
    parts = []
    for (ds, _setting, method), group in df.groupby(["dataset", "setting", "method"], sort=False):
        if method in MIDPOINT_METHODS:
            valid = group.copy()
            valid["op_point_value"] = numeric_or_nan(valid["op_point_value"])
            midpoint = valid[valid["op_point_value"] == MIDPOINT_OP_VALUE]
            if not midpoint.empty:
                parts.append(midpoint.head(1))
                continue
            valid = valid.dropna(subset=["op_point_value"]).sort_values("op_point_value")
            parts.append(valid.iloc[[len(valid) // 2]] if not valid.empty else group)
            continue

        if ds != "PANDA" or method not in PANDA_SINGLE_POINT_METHODS:
            parts.append(group)
            continue

        valid = group.copy()
        valid["total_latency_ms"] = numeric_or_nan(valid["total_latency_ms"])
        valid["op_point_value"] = numeric_or_nan(valid["op_point_value"])
        valid = valid.dropna(subset=["total_latency_ms"])
        if valid.empty:
            parts.append(group)
            continue
        parts.append(
            valid.sort_values(
                ["total_latency_ms", "op_point_value"],
                ascending=[True, True],
            ).head(1)
        )
    return pd.concat(parts, ignore_index=True) if parts else df


def dataset_title(ds: str) -> str:
    if ds.upper() == "PANDA":
        return f"{ds} - Setting 2 (Panda train+test pool, C=4096/L=3)"
    return f"{ds} - Setting 2 (in-distribution, C=128/L=3)"


def render_one(df_ds: pd.DataFrame, ds: str, out_path: Path) -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(dataset_title(ds), fontsize=14)

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
    apply_axis_margins(axA)
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
    apply_axis_margins(axB)
    axB.set_xlabel("Total latency (ms, log)")
    axB.set_ylabel("Retrieval Recall@K (%)")
    axB.set_title("Panel B — Retrieval effectiveness vs efficiency")
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
    df = select_display_points(select_best_gr_baselines(pd.read_csv(args.csv)))
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            print(f"skip {ds}: no rows", flush=True)
            continue
        render_one(sub, ds, args.out_dir / f"{ds.lower()}_panel_AB.png")


if __name__ == "__main__":
    main()
