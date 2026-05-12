"""Checkpoint loading helpers for XPool inference scripts."""

import os

import torch


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_state_dict_compat(model, checkpoint_or_path, *, map_location="cpu"):
    """Load an XPool checkpoint while tolerating CLIP position-id buffer drift."""
    if isinstance(checkpoint_or_path, (str, os.PathLike)):
        checkpoint = torch.load(checkpoint_or_path, map_location=map_location)
    else:
        checkpoint = checkpoint_or_path

    state_dict = _checkpoint_state_dict(checkpoint)
    model_keys = set(model.state_dict().keys())
    dropped_keys = sorted(
        key
        for key in state_dict.keys()
        if key not in model_keys and key.endswith(".position_ids")
    )
    if dropped_keys:
        state_dict = {
            key: value for key, value in state_dict.items() if key not in dropped_keys
        }
        print(
            "Dropped checkpoint-only CLIP position_id buffers: "
            + ", ".join(dropped_keys)
        )

    return model.load_state_dict(state_dict)
