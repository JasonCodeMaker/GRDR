import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer.trainer import OurTrainer


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


if __name__ == "__main__":
    test_route_agreement_prefers_matching_routes()
    test_route_agreement_treats_duplicate_captions_as_multi_positive()
    test_route_agreement_rejects_shape_mismatch()
    print("route agreement loss tests passed")
