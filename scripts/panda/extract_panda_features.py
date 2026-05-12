#!/usr/bin/env python3
"""Extract InternVideo2-1B video + text features for Panda-70M-10M."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

GRDR_ROOT = Path(__file__).resolve().parents[2]
IV2_ROOT = Path('/home/uqzzha35/Project/SemanticID/MM-SemanticTVR/data_process/InternVideo2')
sys.path.insert(0, str(IV2_ROOT))

from utils.config import Config, eval_dict_leaf  # noqa: E402
from tasks.extract_features import load_model_and_tokenizer  # noqa: E402

PANDA_FRAMES_ROOT = GRDR_ROOT / 'dataset' / 'Panda-70M-10M' / 'panda_10m_frames'
PANDA_ANNO_ROOT = GRDR_ROOT / 'dataset' / 'Panda-70M-10M' / 'video_retreival_caption'
OUTPUT_ROOT = GRDR_ROOT / 'dataset' / 'features' / 'InternVideo2' / 'panda'

NUM_FRAMES = 4
IMAGE_RES = 224
IV2_MEAN = (0.485, 0.456, 0.406)
IV2_STD = (0.229, 0.224, 0.225)

SAVE_EVERY_BATCHES = 100


def build_transform():
    """Match get_test_transform() for InternVideo2 vision tower."""
    return transforms.Compose([
        transforms.Resize((IMAGE_RES, IMAGE_RES), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(IV2_MEAN, IV2_STD),
    ])


class PandaFrameDataset(Dataset):
    """Loads 4 JPEG frames per clip and applies the InternVideo2 test transform."""

    def __init__(self, clip_ids, frames_root, transform, num_frames=4):
        self.clip_ids = list(clip_ids)
        self.frames_root = Path(frames_root)
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.clip_ids)

    def _sample_indices(self, vlen):
        if vlen >= self.num_frames:
            seg = vlen / self.num_frames
            return [int(seg * (i + 0.5)) for i in range(self.num_frames)]
        return list(range(vlen)) + [vlen - 1] * (self.num_frames - vlen)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        clip_dir = self.frames_root / clip_id
        try:
            frame_files = sorted(
                f for f in os.listdir(clip_dir) if f.startswith('frame_')
            )
            if not frame_files:
                raise RuntimeError(f'No frames in {clip_dir}')
            indices = self._sample_indices(len(frame_files))
            frames = []
            for i in indices:
                img = Image.open(clip_dir / frame_files[i]).convert('RGB')
                frames.append(np.asarray(img))
            frames_t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
            return self.transform(frames_t), clip_id
        except Exception as e:
            print(f'[skip] {clip_id}: {type(e).__name__}: {e}', flush=True)
            return None, clip_id


def _skip_none_collate(batch):
    """Drop None samples (failed clips) and stack the rest."""
    batch = [(f, c) for f, c in batch if f is not None]
    if not batch:
        return None, []
    frames = torch.stack([b[0] for b in batch], dim=0)
    ids = [b[1] for b in batch]
    return frames, ids


def load_records(split, annotation_root):
    """Return [(clip_id, caption), ...] with stripped split-prefixed ids."""
    if split == 'train_pesudo':
        path = annotation_root / 'panda_10m_ret_train_addition.json'
    else:
        path = annotation_root / f'panda_10m_ret_{split}.json'
    with open(path) as f:
        anns = json.load(f)
    records = []
    for ann in anns:
        vid = ann['video']
        if vid.endswith('.mp4'):
            vid = vid[:-4]
        if '/' in vid:
            vid = vid.split('/')[-1]
        records.append((vid, ann['caption']))
    return records


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def load_partial(path):
    if not path.exists():
        return {}
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f'[resume] {path.name}: {len(data)} entries')
    return data


def extract_video(model, clip_ids, split, args, device, use_bf16):
    out_path = args.output_root / f'video_embeddings_{split}.pkl'
    partial = args.output_root / f'video_embeddings_{split}.partial.pkl'
    if out_path.exists():
        print(f'[video] {out_path.name} exists, skipping')
        return

    features = load_partial(partial)
    todo = [cid for cid in clip_ids if cid not in features]
    if not todo:
        save_pickle(features, out_path)
        partial.unlink(missing_ok=True)
        print(f'[video] {len(features)} already cached -> {out_path}')
        return

    print(f'[video] split={split} todo={len(todo)} done={len(features)}')
    frames_root = args.frames_root / split
    dataset = PandaFrameDataset(todo, frames_root, build_transform(), NUM_FRAMES)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=_skip_none_collate,
    )

    cast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model.eval()
    batches_since_save = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=cast_dtype):
        for frames, ids in tqdm(loader, desc=f'video/{split}'):
            if frames is None or not ids:
                continue
            frames = frames.to(device, non_blocking=True)
            _, pooled = model.encode_vision(frames, test=True)
            proj = model.vision_proj(pooled).cpu().float().numpy()
            for j, cid in enumerate(ids):
                features[cid] = proj[j]
            batches_since_save += 1
            if batches_since_save >= SAVE_EVERY_BATCHES:
                save_pickle(features, partial)
                batches_since_save = 0

    save_pickle(features, out_path)
    partial.unlink(missing_ok=True)
    print(f'[video] wrote {len(features)} -> {out_path}')


def extract_text(model, tokenizer, records, split, args, device, use_bf16):
    out_path = args.output_root / f'text_embeddings_{split}.pkl'
    merge_existing = getattr(args, 'merge_existing', False)
    if out_path.exists() and not merge_existing:
        print(f'[text] {out_path.name} exists, skipping')
        return

    captions, keys, counts = [], [], {}
    for vid, cap in records:
        idx = counts.get(vid, 0)
        counts[vid] = idx + 1
        keys.append(f'{vid}_{idx}')
        captions.append(cap)

    if merge_existing and out_path.exists():
        with open(out_path, 'rb') as f:
            features = pickle.load(f)
        print(f'[text/merge] seeded with {len(features)} existing entries from {out_path.name}')
    else:
        features = {}
    todo_pairs = [(k, c) for k, c in zip(keys, captions) if k not in features]
    if not todo_pairs:
        print(f'[text] all {len(features)} keys already present, nothing to do')
        return
    todo_keys = [p[0] for p in todo_pairs]
    todo_captions = [p[1] for p in todo_pairs]
    print(f'[text] split={split} todo={len(todo_keys)} done={len(features)}')

    cast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=cast_dtype):
        for i in tqdm(range(0, len(todo_captions), args.text_batch_size), desc=f'text/{split}'):
            batch = todo_captions[i:i + args.text_batch_size]
            batch_keys = todo_keys[i:i + args.text_batch_size]
            inputs = tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=args.max_txt_l,
                return_tensors='pt',
            ).to(device)
            text_feats = model.encode_text(inputs)[0]
            proj = model.text_proj(text_feats[:, 0]).cpu().float().numpy()
            for j, k in enumerate(batch_keys):
                features[k] = proj[j]

    save_pickle(features, out_path)
    print(f'[text] wrote {len(features)} -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test', 'train_pesudo'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', default='both', choices=['both', 'video', 'text'])
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--text_batch_size', type=int, default=256)
    parser.add_argument('--max_txt_l', type=int, default=40)
    parser.add_argument('--frames-root', type=Path, default=PANDA_FRAMES_ROOT)
    parser.add_argument('--annotation-root', type=Path, default=PANDA_ANNO_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--merge-existing', dest='merge_existing', action='store_true',
                        help='Text path: load existing pkl as seed, only compute missing keys, write merged pkl')
    parser.add_argument('--include-keys-file', type=Path, default=None,
                        help='Optional text file (one clip_id per line). If set, video and text paths process only listed clip_ids.')
    args = parser.parse_args()
    args.frames_root = args.frames_root.resolve()
    args.annotation_root = args.annotation_root.resolve()
    args.output_root = args.output_root.resolve()

    # Caller is expected to set CUDA_VISIBLE_DEVICES before python starts
    # (the accompanying shell wrapper does this). --gpu_id is the logical
    # index inside that mask and defaults to 0.
    device = torch.device(f'cuda:{args.gpu_id}')

    config_path = IV2_ROOT / 'scripts/evaluation/stage2/zero_shot/1B/config_msrvtt.py'
    config = Config.from_file(str(config_path))
    config = eval_dict_leaf(config)

    print(f'[model] loading InternVideo2-1B from {args.checkpoint}')
    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)

    records = load_records(args.split, args.annotation_root)
    if args.include_keys_file is not None:
        keep = {ln.strip() for ln in args.include_keys_file.read_text().splitlines() if ln.strip()}
        records = [r for r in records if r[0] in keep]
        print(f'[data] include-keys-file filter: kept {len(records)} records intersecting {len(keep)} clip_ids')
    clip_ids = list(dict.fromkeys(r[0] for r in records))
    print(f'[data] split={args.split} captions={len(records)} unique_clips={len(clip_ids)}')

    use_bf16 = config.get('use_bf16', True)

    if args.mode in ('both', 'video'):
        extract_video(model, clip_ids, args.split, args, device, use_bf16)
    if args.mode in ('both', 'text'):
        extract_text(model, tokenizer, records, args.split, args, device, use_bf16)


if __name__ == '__main__':
    main()
