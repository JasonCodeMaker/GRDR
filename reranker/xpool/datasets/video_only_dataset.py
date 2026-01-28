"""
VideoOnlyDataset: Loads only video frames (no text) for expanded pool evaluation.

Used when evaluating with an expanded search pool (train + test videos)
while keeping test queries and ground truth unchanged.
"""

import os
import torch
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


def default_path_fn(videos_dir, vid, video_ext):
    """Default path builder: simple concatenation."""
    return os.path.join(videos_dir, vid + video_ext)


class VideoOnlyDataset(Dataset):
    """
    Dataset that loads only video frames without text captions.

    Used for loading additional videos (e.g., training set) to expand
    the search pool during evaluation.
    """

    def __init__(self, config: Config, video_ids: list, videos_dir: str,
                 img_transforms=None, video_ext: str = '.mp4', path_fn=None):
        """
        Args:
            config: Configuration object with num_frames, video_sample_type
            video_ids: List of video IDs to load
            videos_dir: Directory containing video files
            img_transforms: Image transforms to apply
            video_ext: Video file extension (e.g., '.mp4', '.avi')
            path_fn: Optional function(videos_dir, vid, video_ext) -> video_path
                     For datasets with complex path structures (e.g., LSMDC)
        """
        self.config = config
        self.video_ids = video_ids
        self.videos_dir = videos_dir
        self.img_transforms = img_transforms
        self.video_ext = video_ext
        self.path_fn = path_fn if path_fn is not None else default_path_fn

    def __getitem__(self, index):
        vid = self.video_ids[index]
        video_path = self.path_fn(self.videos_dir, vid, self.video_ext)

        imgs, _ = VideoCapture.load_frames_from_video(
            video_path,
            self.config.num_frames,
            self.config.video_sample_type
        )

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': vid,
            'video': imgs,
        }

    def __len__(self):
        return len(self.video_ids)
