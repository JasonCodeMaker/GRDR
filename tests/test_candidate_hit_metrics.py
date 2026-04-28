import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer.evaluator import compute_candidate_hit_metrics, expand_sid_predictions_to_videos


def make_distractors(prefix, count):
    return [f"{prefix}{idx:03d}" for idx in range(count)]


def test_candidate_hit_metrics_use_expanded_deduped_video_ranks():
    sid_to_videos = {
        "sid_dup": ["noise0_0", "video_dup_1", "video_dup_2"],
        "sid_rank11_a": make_distractors("rank11_noise", 10),
        "sid_rank11_b": ["video_rank11_3"],
        "sid_rank31_a": make_distractors("rank31_noise", 30),
        "sid_rank31_b": ["video_rank31_4"],
        "sid_sparse": ["only_candidate"],
    }

    assert expand_sid_predictions_to_videos(["sid_dup"], sid_to_videos) == ["noise0", "video_dup"]

    predictions = [
        ["sid_dup"],
        ["sid_rank11_a", "sid_rank11_b"],
        ["sid_rank31_a", "sid_rank31_b"],
        ["sid_sparse"],
    ]
    ground_truth = [
        "video_dup_0",
        "video_rank11_0",
        "video_rank31_0",
        "missing_video_0",
    ]

    metrics = compute_candidate_hit_metrics(predictions, ground_truth, sid_to_videos, ks=(10, 20, 50))

    assert metrics == {
        "CanHit@10": 25.0,
        "CanHit@20": 50.0,
        "CanHit@50": 75.0,
    }


if __name__ == "__main__":
    test_candidate_hit_metrics_use_expanded_deduped_video_ranks()
    print("candidate hit metric tests passed")
