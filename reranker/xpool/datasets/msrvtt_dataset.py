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


class MSRVTTDataset(Dataset):
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
        db_file = 'reranker/xpool/data/MSRVTT/MSRVTT_data.json'
        test_csv = 'reranker/xpool/data/MSRVTT/MSRVTT_JSFUSION_test.csv'

        if config.msrvtt_train_file == '7k':
            train_csv = 'reranker/xpool/data/MSRVTT/MSRVTT_train.7k.csv'
        else:
            train_csv = 'reranker/xpool/data/MSRVTT/MSRVTT_train.9k.csv'

        self.db = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(test_csv)
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
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence

        return video_path, caption, vid

    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])

            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)
    
    def _generate_candidate_mask(self, candidate_file, extra_vid_ids=None):
        """
        Generate boolean mask for candidate reranking constraints.

        Args:
            candidate_file: Path to JSON file with candidate constraints
            extra_vid_ids: Optional list of additional video IDs (e.g., train videos)
                          to include in the candidate pool for expanded evaluation

        Returns:
            candidate_mask: Boolean tensor [num_test_vids, max_text_per_vid, num_all_vids]
                           where True indicates valid query-video pairs.
                           When extra_vid_ids is provided, num_all_vids = num_test_vids + len(extra_vid_ids)
        """
        # Load candidate constraints
        with open(candidate_file, 'r') as f:
            candidate_data = json.load(f)

        results = candidate_data['results']

        # Build query_text -> candidate_videos mapping
        query_to_candidates = {}
        for result in results:
            query_text = result['query_text']
            candidates = result['candidates']
            query_to_candidates[query_text] = set(candidates)

        # Get all video IDs and query texts from test_df
        all_vid_ids = self.test_df['video_id'].tolist()
        all_texts = self.test_df['sentence'].tolist()

        # Build unique video ID mapping using SAME logic as generate_embeds_per_video_id
        # This ensures mask ordering matches similarity matrix ordering exactly
        text_embeds_per_video_id = {}
        for idx, v_id in enumerate(all_vid_ids):
            if v_id not in text_embeds_per_video_id:
                text_embeds_per_video_id[v_id] = []

        # Use the keys in the same order as generate_embeds_per_video_id will use -- Checked Correct
        unique_vid_ids = list(text_embeds_per_video_id.keys())
        num_test_vids = len(unique_vid_ids)

        # Extend with extra videos (e.g., train set) for expanded pool evaluation
        if extra_vid_ids is not None:
            test_set = set(unique_vid_ids)
            for vid in extra_vid_ids:
                if vid not in test_set:
                    unique_vid_ids.append(vid)

        vid_to_unique_idx = {vid: idx for idx, vid in enumerate(unique_vid_ids)}
        num_all_vids = len(unique_vid_ids)

        # Build text list for each video in the SAME order as generate_embeds_per_video_id
        text_list_per_video_id = {}
        for idx, v_id in enumerate(all_vid_ids):
            if v_id in text_list_per_video_id:
                text_list_per_video_id[v_id].append(all_texts[idx])
            else:
                text_list_per_video_id[v_id] = [all_texts[idx]]

        # Calculate max_text_per_vid from the actual text lists
        max_text_per_vid = max(len(text_list) for text_list in text_list_per_video_id.values()) if text_list_per_video_id else 1

        # Initialize mask as all False (no valid pairs)
        # Shape: [num_test_vids, max_text_per_vid, num_all_vids]
        # Note: first dimension is num_test_vids (queries only from test set)
        #       third dimension is num_all_vids (candidates from test + extra)
        candidate_mask = torch.zeros(num_test_vids, max_text_per_vid, num_all_vids, dtype=torch.bool)

        # Fill mask using the same iteration order as generate_embeds_per_video_id
        # Only iterate over test videos (queries come from test set only)
        for query_vid_idx in range(num_test_vids):
            query_vid_id = unique_vid_ids[query_vid_idx]
            text_list = text_list_per_video_id[query_vid_id]

            for text_pos, query_text in enumerate(text_list):
                # Find valid candidate videos for this specific query text
                if query_text in query_to_candidates:
                    candidates = query_to_candidates[query_text]

                    for candidate_vid in candidates:
                        if candidate_vid in vid_to_unique_idx:
                            candidate_unique_idx = vid_to_unique_idx[candidate_vid]
                            candidate_mask[query_vid_idx, text_pos, candidate_unique_idx] = True

        return candidate_mask # Checked Correct
