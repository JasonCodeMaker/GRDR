import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class LSMDCDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        
        self.clip2caption = {}
        if split_type == 'train':
            train_file = 'reranker/xpool/data/LSMDC/LSMDC16_annos_training.csv'
            self._compute_clip2caption(train_file)
        else:
            test_file = 'reranker/xpool/data/LSMDC/LSMDC16_challenge_1000_publictect.csv'
            self._compute_clip2caption(test_file)

        # candidate_mask is generated in test.py after expanded_pool decision
        self.candidate_mask = None


    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
        }

    
    def __len__(self):
        return len(self.clip2caption)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        clip_id = list(self.clip2caption.keys())[index]
        caption = self.clip2caption[clip_id]
        clip_prefix = clip_id.split('.')[0][:-3]
        video_path = os.path.join(self.videos_dir, clip_prefix, clip_id + '.avi')

        return video_path, caption, clip_id

            
    def _compute_clip2caption(self, csv_file):
        with open(csv_file, 'r') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split("\t")
                assert len(line_split) == 6
                clip_id, _, _, _, _, caption = line_split
                if clip_id == '1012_Unbreakable_00.05.16.065-00.05.21.941':
                    continue
                self.clip2caption[clip_id] = caption

    def _generate_candidate_mask(self, candidate_file, extra_vid_ids=None):
        """
        Generate boolean mask for candidate reranking constraints.

        Args:
            candidate_file: Path to JSON file with candidate constraints
            extra_vid_ids: Optional list of additional clip IDs (e.g., train clips)
                          to include in the candidate pool for expanded evaluation

        Returns:
            candidate_mask: Boolean tensor [num_test_clips, 1, num_all_clips]
                           where True indicates valid query-video pairs.
                           When extra_vid_ids is provided, num_all_clips = num_test_clips + len(extra_vid_ids)
        """
        # Load candidate constraints
        with open(candidate_file, 'r') as f:
            candidate_data = json.load(f)

        results = candidate_data['results']

        # Build query_text -> candidate_clips mapping
        query_to_candidates = {}
        for result in results:
            query_text = result['query_text']
            candidates = result['candidates']
            query_to_candidates[query_text] = set(candidates)

        # Get all clip IDs and captions from clip2caption
        # Use list() to get consistent iteration order (same as __getitem__)
        all_clip_ids = list(self.clip2caption.keys())
        all_texts = [self.clip2caption[clip_id] for clip_id in all_clip_ids]
        num_test_clips = len(all_clip_ids)

        # Extend with extra clips (e.g., train set) for expanded pool evaluation
        if extra_vid_ids is not None:
            test_set = set(all_clip_ids)
            for clip_id in extra_vid_ids:
                if clip_id not in test_set:
                    all_clip_ids.append(clip_id)

        # Build unique clip ID mapping
        clip_to_idx = {clip_id: idx for idx, clip_id in enumerate(all_clip_ids)}
        num_all_clips = len(all_clip_ids)

        max_text_per_clip = 1  # LSMDC has 1:1 clip-caption mapping

        # Initialize mask as all False
        # Shape: [num_test_clips, 1, num_all_clips]
        candidate_mask = torch.zeros(num_test_clips, max_text_per_clip, num_all_clips, dtype=torch.bool)

        # Fill mask - only iterate over test clips (queries come from test set only)
        for query_idx in range(num_test_clips):
            clip_id = list(self.clip2caption.keys())[query_idx]
            query_text = all_texts[query_idx]
            if query_text in query_to_candidates:
                candidates = query_to_candidates[query_text]
                for candidate_clip in candidates:
                    if candidate_clip in clip_to_idx:
                        candidate_idx = clip_to_idx[candidate_clip]
                        candidate_mask[query_idx, 0, candidate_idx] = True

        return candidate_mask
