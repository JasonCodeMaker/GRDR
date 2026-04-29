import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer.evaluator import (
    compute_candidate_hit_metrics,
    compute_detailed_metrics,
    expand_sid_predictions_to_videos,
)


def make_distractors(prefix, count):
    return [f"{prefix}{idx:03d}" for idx in range(count)]


def test_candidate_hit_metrics_use_expanded_deduped_video_ranks():
    sid_to_videos = {
        "sid_dup": ["noise0_0", "video_dup_1", "video_dup_2"],
        "sid_rank31_a": make_distractors("rank31_noise", 30),
        "sid_rank31_b": ["video_rank31_4"],
        "sid_rank71_a": make_distractors("rank71_noise", 70),
        "sid_rank71_b": ["video_rank71_5"],
        "sid_sparse": ["only_candidate"],
    }

    assert expand_sid_predictions_to_videos(["sid_dup"], sid_to_videos) == ["noise0", "video_dup"]

    predictions = [
        ["sid_dup"],
        ["sid_rank31_a", "sid_rank31_b"],
        ["sid_rank71_a", "sid_rank71_b"],
        ["sid_sparse"],
    ]
    ground_truth = [
        "video_dup_0",
        "video_rank31_0",
        "video_rank71_0",
        "missing_video_0",
    ]

    metrics = compute_candidate_hit_metrics(predictions, ground_truth, sid_to_videos)

    assert metrics == {
        "CanHit@20": 25.0,
        "CanHit@50": 50.0,
        "CanHit@100": 75.0,
    }


def test_export_metrics_report_candidate_hits_not_sid_recall():
    sid_to_videos = {
        "sid_hit": ["video_hit_0"],
        "sid_late_a": make_distractors("late_noise", 30),
        "sid_late_b": ["video_late_0"],
        "sid_miss": ["noise"],
    }
    predictions = [
        ["sid_hit"],
        ["sid_late_a", "sid_late_b"],
        ["sid_miss"],
    ]
    ground_truth = ["video_hit_0", "video_late_0", "missing_0"]
    results = [
        {"num_candidates": 1},
        {"num_candidates": 31},
        {"num_candidates": 1},
    ]

    metrics = compute_detailed_metrics(
        results,
        predictions,
        ground_truth,
        sid_to_videos,
        total_time=1.5,
        num_queries=3,
        num_candidates=20,
    )

    assert "Recall@10" not in metrics
    assert abs(metrics["CanHit@20"] - (100 / 3)) < 1e-9
    assert abs(metrics["CanHit@50"] - (200 / 3)) < 1e-9
    assert abs(metrics["CanHit@100"] - (200 / 3)) < 1e-9
    assert metrics["avg_candidates_per_query"] == 11.0


if __name__ == "__main__":
    test_candidate_hit_metrics_use_expanded_deduped_video_ranks()
    test_export_metrics_report_candidate_hits_not_sid_recall()
    print("candidate hit metric tests passed")
