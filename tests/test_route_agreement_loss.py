import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer.trainer import OurTrainer


def one_hot_logits(indices, num_codes=3, high=10.0, low=-10.0):
    logits = torch.full((len(indices), 1, num_codes), low)
    for row, idx in enumerate(indices):
        logits[row, 0, idx] = high
    return logits


def test_route_agreement_prefers_matching_routes():
    query_logits = torch.tensor([
        [[10.0, -10.0], [10.0, -10.0]],
        [[-10.0, 10.0], [-10.0, 10.0]],
    ])
    video_logits = query_logits.clone()

    matching_loss = OurTrainer.compute_route_agreement_loss(
        query_logits,
        video_logits,
        ["video1_0", "video2_0"],
    )
    swapped_loss = OurTrainer.compute_route_agreement_loss(
        query_logits,
        video_logits.flip(0),
        ["video1_0", "video2_0"],
    )

    assert matching_loss.item() < 1e-3
    assert swapped_loss.item() > 10.0


def test_route_agreement_treats_duplicate_captions_as_multi_positive():
    query_logits = torch.tensor([
        [[10.0, -10.0]],
        [[-10.0, 10.0]],
    ])
    video_logits = query_logits.clone()

    loss = OurTrainer.compute_route_agreement_loss(
        query_logits,
        video_logits,
        ["video1_0", "video1_1"],
    )

    assert loss.item() < 1e-3


def test_route_agreement_rejects_shape_mismatch():
    query_logits = torch.zeros(2, 2, 3)
    video_logits = torch.zeros(2, 1, 3)

    try:
        OurTrainer.compute_route_agreement_loss(query_logits, video_logits)
    except ValueError:
        return

    raise AssertionError("shape mismatch did not raise")


def test_bucket_route_loss_prefers_compact_positive_route():
    video_logits = one_hot_logits([0, 1, 2])
    compact_query = one_hot_logits([1, 1, 2])
    broad_query = one_hot_logits([0, 0, 2])
    video_ids = ["video1_0", "video1_1", "video2_0"]
    bucket_sizes = torch.tensor([100.0, 1.0, 1.0])

    compact_loss = OurTrainer.compute_bucket_route_loss(
        compact_query,
        video_logits,
        video_ids=video_ids,
        bucket_sizes=bucket_sizes,
        gamma=2.0,
    )
    broad_loss = OurTrainer.compute_bucket_route_loss(
        broad_query,
        video_logits,
        video_ids=video_ids,
        bucket_sizes=bucket_sizes,
        gamma=2.0,
    )

    assert compact_loss.item() < broad_loss.item()


def test_video_rank_loss_penalizes_large_positive_bucket():
    query_logits = one_hot_logits([0, 1])
    video_logits = one_hot_logits([0, 1])
    video_ids = ["video1_0", "video2_0"]
    bucket_sizes = torch.tensor([100.0, 1.0])

    no_penalty = OurTrainer.compute_video_rank_loss(
        query_logits,
        video_logits,
        video_ids=video_ids,
        bucket_sizes=bucket_sizes,
        beta=0.0,
    )
    with_penalty = OurTrainer.compute_video_rank_loss(
        query_logits,
        video_logits,
        video_ids=video_ids,
        bucket_sizes=bucket_sizes,
        beta=2.0,
    )

    assert with_penalty.item() > no_penalty.item()


def test_expanded_size_loss_is_larger_for_large_bucket_mass():
    video_logits = one_hot_logits([0, 1])
    broad_query = one_hot_logits([0, 0])
    compact_query = one_hot_logits([1, 1])
    bucket_sizes = torch.tensor([100.0, 1.0])

    broad_loss = OurTrainer.compute_expanded_size_loss(
        broad_query,
        video_logits,
        bucket_sizes=bucket_sizes,
    )
    compact_loss = OurTrainer.compute_expanded_size_loss(
        compact_query,
        video_logits,
        bucket_sizes=bucket_sizes,
    )

    assert broad_loss.item() > compact_loss.item()


if __name__ == "__main__":
    test_route_agreement_prefers_matching_routes()
    test_route_agreement_treats_duplicate_captions_as_multi_positive()
    test_route_agreement_rejects_shape_mismatch()
    test_bucket_route_loss_prefers_compact_positive_route()
    test_video_rank_loss_penalizes_large_positive_bucket()
    test_expanded_size_loss_is_larger_for_large_bucket_mass()
    print("route agreement loss tests passed")
