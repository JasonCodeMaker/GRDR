import os
import torch
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
        # candidate_mask is generated in test.py after the expanded_pool decision
        self.candidate_mask = None
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

    def _generate_candidate_mask(self, candidate_file, extra_vid_ids=None):
        """Boolean mask [num_test, 1, num_all] True only at each test query's stage-1 candidates (Panda test is 1:1)."""
        def bare(v):
            return strip_media_extension(v).split('/')[-1]

        with open(candidate_file) as f:
            candidate_data = json.load(f)
        # Match by ground-truth video id (robust to duplicate captions); candidate ids are bare.
        gt_to_candidates = {r['ground_truth_video_id']: set(r['candidates']) for r in candidate_data['results']}

        # Column order MUST mirror the trainer's video-embedding matrix: test entries (in order),
        # then expanded-pool extra_vid_ids (deduped against test, as in the other dataset masks).
        test_ids = [bare(e['video']) for e in self.entries]
        col_ids = list(test_ids)
        if extra_vid_ids is not None:
            test_set = set(test_ids)
            for v in extra_vid_ids:
                vb = bare(v)
                if vb not in test_set:
                    col_ids.append(vb)
        col_index = {v: i for i, v in enumerate(col_ids)}
        num_test, num_all = len(test_ids), len(col_ids)

        candidate_mask = torch.zeros(num_test, 1, num_all, dtype=torch.bool)
        for i, vid in enumerate(test_ids):
            for c in gt_to_candidates.get(vid, ()):  # bare candidate id -> column
                j = col_index.get(c)
                if j is not None:
                    candidate_mask[i, 0, j] = True
        return candidate_mask
