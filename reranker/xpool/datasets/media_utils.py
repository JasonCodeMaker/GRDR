import os
from typing import Optional


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def strip_media_extension(video_id: str) -> str:
    base, ext = os.path.splitext(video_id)
    if ext.lower() in VIDEO_EXTENSIONS:
        return base
    return video_id


def _dedupe_preserve_order(paths):
    seen = set()
    ordered = []
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def resolve_existing_path(*candidates: str) -> str:
    ordered = _dedupe_preserve_order(candidates)
    for candidate in ordered:
        if os.path.exists(candidate):
            return candidate
    if not ordered:
        raise ValueError("resolve_existing_path requires at least one candidate")
    return ordered[0]


def resolve_media_root(dataset_name: str, media_root: str) -> str:
    if dataset_name == 'MSRVTT':
        candidates = [
            media_root.replace('MSRVTT_Videos', 'MSRVTT_Frames'),
            media_root,
        ]
    elif dataset_name == 'ACTNET':
        candidates = [
            media_root.replace('Activity_Videos', 'Activity_Frames_224x224'),
            media_root.replace('Activity_Videos', 'Activity_Frames'),
            media_root,
        ]
    elif dataset_name == 'DIDEMO':
        candidates = [
            media_root.replace(os.path.join('test', 'test_videos'), ''),
            media_root.replace(os.path.join('train', 'videos'), ''),
            media_root,
        ]
    elif dataset_name == 'LSMDC':
        candidates = [
            media_root.replace('LSMDC_Videos', 'LSMDC_Frames_224x224'),
            media_root.replace('LSMDC_Videos', 'LSMDC_Frames_256'),
            media_root,
        ]
    else:
        candidates = [media_root]

    return resolve_existing_path(*candidates)


def resolve_media_path(dataset_name: str, media_root: str, video_id: str, split_type: Optional[str] = None) -> str:
    media_root = resolve_media_root(dataset_name, media_root)
    normalized_id = strip_media_extension(video_id)
    candidates = []

    if dataset_name == 'MSRVTT':
        candidates.extend([
            os.path.join(media_root, normalized_id + '.mp4'),
            os.path.join(media_root, normalized_id),
        ])
    elif dataset_name == 'ACTNET':
        candidates.extend([
            os.path.join(media_root, video_id),
            os.path.join(media_root, normalized_id),
            os.path.join(media_root, normalized_id + '.mp4'),
        ])
    elif dataset_name == 'DIDEMO':
        split = 'train' if split_type == 'train' else 'test'
        split_roots = []
        if split == 'train':
            split_roots.extend([
                os.path.join(media_root, 'train_frame_224x224'),
                os.path.join(media_root, 'train_frame'),
                os.path.join(media_root, 'train', 'videos'),
            ])
        else:
            split_roots.extend([
                os.path.join(media_root, 'test_frame_224x224'),
                os.path.join(media_root, 'test_frame'),
                os.path.join(media_root, 'test', 'test_videos'),
            ])
        split_roots.append(media_root)

        for root in split_roots:
            candidates.extend([
                os.path.join(root, video_id),
                os.path.join(root, normalized_id),
                os.path.join(root, normalized_id + '.mp4'),
            ])
    elif dataset_name == 'LSMDC':
        clip_prefix = normalized_id.split('.')[0][:-3]
        candidates.extend([
            os.path.join(media_root, clip_prefix, normalized_id + '.avi'),
            os.path.join(media_root, clip_prefix, normalized_id),
        ])
    elif dataset_name == 'PANDA':
        # video_id is e.g. 'train/00000000_-xxx_clip00.mp4'. Frame layout:
        # <videos_dir>/<split>/<video_id_no_ext>/frame_NNN.jpg
        candidates.extend([
            os.path.join(media_root, normalized_id),
            os.path.join(media_root, video_id),
            os.path.join(media_root, normalized_id + '.mp4'),
        ])
    else:
        candidates.extend([
            os.path.join(media_root, video_id),
            os.path.join(media_root, normalized_id),
        ])

    return resolve_existing_path(*candidates)
