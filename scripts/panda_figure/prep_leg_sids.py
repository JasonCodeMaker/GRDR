#!/usr/bin/env python
"""Emit a per-leg TIGER/AVG train sID index truncated to a manifest's first-d distractors.

The full train sID index is a dict {video_id(bare): [code tokens]}. A distractor
manifest lists the first-d distractor video ids (with a 'train/' prefix that the sID
keys do not carry). This writes a new dict containing only those d videos, preserving
manifest order, so the MM-SemanticTVR eval pool = test sIDs + these d train sIDs.
"""
import argparse
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full_train_index", required=True, help="full train sID index JSON (dict)")
    p.add_argument("--manifest", required=True, help="panda_pool_d<D>.json with video_ids")
    p.add_argument("--out", required=True, help="output truncated train sID index JSON")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.full_train_index) as f:
        full = json.load(f)
    with open(args.manifest) as f:
        manifest = json.load(f)

    leg = {}
    missing = 0
    for vid in manifest["video_ids"]:
        bare = vid.split("/", 1)[1] if "/" in vid else vid
        if bare in full:
            leg[bare] = full[bare]
        else:
            missing += 1

    with open(args.out, "w") as f:
        json.dump(leg, f)
    print(f"wrote {args.out}: {len(leg)} train sIDs "
          f"(manifest n_distractors={manifest.get('n_distractors')}, missing={missing})")


if __name__ == "__main__":
    main()
