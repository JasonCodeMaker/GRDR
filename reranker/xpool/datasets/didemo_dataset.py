import os
import torch
try:
    import ujson as json
except ImportError:
    import json
from collections import defaultdict
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture
from modules.basic_utils import load_json


class DiDeMoDataset(Dataset):
    """
    DiDeMo dataset for video-text retrieval.

    Annotation format: [{video: "xxx.mp4", caption: ["cap1", "cap2", ...]}]
    Video path: Split-specific directories
        - Train: {videos_dir}/train/videos/{video_id}
        - Test: {videos_dir}/test/test_videos/{video_id}
    """

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.img_transforms = img_transforms
        self.split_type = split_type

        # Set up split-specific video directory
        if split_type == 'train':
            self.video_root = os.path.join(config.videos_dir, 'train', 'videos')
            anno_file = 'reranker/xpool/data/DIDEMO/didemo_ret_train.json'
        else:
            self.video_root = os.path.join(config.videos_dir, 'test', 'test_videos')
            anno_file = 'reranker/xpool/data/DIDEMO/didemo_ret_test.json'

        self.annotations = load_json(anno_file)

        # Build vid2caption mapping
        self._build_vid2caption()

        # Build (video_id, caption) pairs
        self._construct_all_pairs(split_type)

        # candidate_mask is generated in test.py after expanded_pool decision
        self.candidate_mask = None

    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(
            video_path,
            self.config.num_frames,
            self.config.video_sample_type
        )

        # Apply image transforms
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
        }

    def __len__(self):
        return len(self.all_pairs)

    def _get_vidpath_and_caption_by_index(self, index):
        """Get video path and caption for the given index."""
        vid, caption = self.all_pairs[index]
        # video_id already includes .mp4 suffix in annotation
        video_path = os.path.join(self.video_root, vid)
        # Remove .mp4 suffix for video_id used in evaluation
        video_id = vid.replace('.mp4', '')
        return video_path, caption, video_id

    def _build_vid2caption(self):
        """Build video_id to captions mapping from JSON annotations."""
        self.vid2caption = defaultdict(list)
        for item in self.annotations:
            vid = item['video']  # e.g., "xxx.mp4"
            captions = item['caption']  # List of captions
            # Expand list captions
            for cap in captions:
                cap_text = cap.strip() if isinstance(cap, str) else cap
                self.vid2caption[vid].append(cap_text)

    def _construct_all_pairs(self, split_type):
        """Construct all (video_id, caption) pairs for iteration."""
        self.all_pairs = []
        if split_type == 'train':
            for vid, captions in self.vid2caption.items():
                for caption in captions:
                    self.all_pairs.append([vid, caption])
        else:
            for vid, captions in self.vid2caption.items():
                # Concatenate all captions as one string, separated by a space
                concat_caption = " ".join([cap.strip() if isinstance(cap, str) else str(cap) for cap in captions])
                self.all_pairs.append([vid, concat_caption])

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

        # Get all video IDs and query texts from pairs
        # Note: DiDeMo stores IDs with .mp4 suffix internally, normalize to match candidate JSON
        all_vid_ids = [pair[0].replace('.mp4', '') for pair in self.all_pairs]
        all_texts = [pair[1] for pair in self.all_pairs]

        # Build unique video ID mapping using SAME logic as generate_embeds_per_video_id
        text_embeds_per_video_id = {}
        for idx, v_id in enumerate(all_vid_ids):
            if v_id not in text_embeds_per_video_id:
                text_embeds_per_video_id[v_id] = []

        # Use the keys in the same order
        unique_vid_ids = list(text_embeds_per_video_id.keys())
        num_test_vids = len(unique_vid_ids)

        # Extend with extra videos (e.g., train set) for expanded pool evaluation
        if extra_vid_ids is not None:
            test_set = set(unique_vid_ids)
            for vid in extra_vid_ids:
                # Normalize: remove .mp4 suffix if present
                normalized_vid = vid.replace('.mp4', '') if vid.endswith('.mp4') else vid
                if normalized_vid not in test_set:
                    unique_vid_ids.append(normalized_vid)

        vid_to_unique_idx = {vid: idx for idx, vid in enumerate(unique_vid_ids)}
        num_all_vids = len(unique_vid_ids)

        # Build text list for each video
        text_list_per_video_id = {}
        for idx, v_id in enumerate(all_vid_ids):
            if v_id in text_list_per_video_id:
                text_list_per_video_id[v_id].append(all_texts[idx])
            else:
                text_list_per_video_id[v_id] = [all_texts[idx]]

        # Calculate max_text_per_vid
        max_text_per_vid = max(len(text_list) for text_list in text_list_per_video_id.values()) if text_list_per_video_id else 1

        # Initialize mask as all False
        # Shape: [num_test_vids, max_text_per_vid, num_all_vids]
        candidate_mask = torch.zeros(num_test_vids, max_text_per_vid, num_all_vids, dtype=torch.bool)

        # Fill mask - only iterate over test videos (queries come from test set only)
        for query_vid_idx in range(num_test_vids):
            query_vid_id = unique_vid_ids[query_vid_idx]
            text_list = text_list_per_video_id[query_vid_id]

            for text_pos, query_text in enumerate(text_list):
                if query_text in query_to_candidates:
                    candidates = query_to_candidates[query_text]

                    for candidate_vid in candidates:
                        if candidate_vid in vid_to_unique_idx:
                            candidate_unique_idx = vid_to_unique_idx[candidate_vid]
                            candidate_mask[query_vid_idx, text_pos, candidate_unique_idx] = True

        return candidate_mask
