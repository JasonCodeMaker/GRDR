import os
import ujson as json
from collections import defaultdict
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture
from datasets.media_utils import resolve_media_path, strip_media_extension


PANDA_NUM_FRAMES = 4  # Panda clips have exactly 4 frames on disk; override config.
PANDA_JSON_ROOT = 'data/panda/video_retreival_caption'


def _panda_train_json_path(config: Config) -> str:
    """Pick the train annotation file: opt-in addition file via --panda_use_pseudo_queries, else legacy 2.15M."""
    if getattr(config, 'panda_use_pseudo_queries', False):
        return os.path.join(PANDA_JSON_ROOT, 'panda_ret_train_addition.json')
    return os.path.join(PANDA_JSON_ROOT, 'panda_ret_train.json')


class PandaDataset(Dataset):
    """Panda-70M-10M for X-Pool. Each row is one (video, caption) pair; GT is 1:1 on both splits."""

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type

        self.config.num_frames = PANDA_NUM_FRAMES

        if split_type == 'train':
            anno_path = _panda_train_json_path(config)
        else:
            anno_path = os.path.join(PANDA_JSON_ROOT, 'panda_ret_test.json')

        with open(anno_path) as f:
            annotations = json.load(f)

        self.entries = annotations
        if split_type == 'train':
            self.vid2caption = defaultdict(list)
            for row in annotations:
                vid = strip_media_extension(row['video'])
                self.vid2caption[vid].append(row['caption'])
            self.all_train_pairs = [(v, c) for v, caps in self.vid2caption.items() for c in caps]

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.entries)

    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames(video_path,
                                              self.config.num_frames,
                                              self.config.video_sample_type)
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)
        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
        }

    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
        else:
            row = self.entries[index]
            vid = row['video']
            caption = row['caption']
        video_path = resolve_media_path('PANDA', self.videos_dir, vid, split_type=self.split_type)
        return video_path, caption, strip_media_extension(vid)
