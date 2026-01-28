import os
import pandas as pd
from config.base_config import Config
from datasets.model_transforms import init_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset
from datasets.msvd_dataset import MSVDDataset
from datasets.lsmdc_dataset import LSMDCDataset
from datasets.actnet_dataset import ActivityNetDataset
from datasets.didemo_dataset import DiDeMoDataset
from datasets.video_only_dataset import VideoOnlyDataset
from torch.utils.data import DataLoader
from modules.basic_utils import load_json, read_lines

class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']

        if config.dataset_name == "MSRVTT":
            if split_type == 'train':
                dataset = MSRVTTDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers)
            else:                
                dataset = MSRVTTDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == "MSVD":
            if split_type == 'train':
                dataset = MSVDDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)
            else:
                dataset = MSVDDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)
            
        elif config.dataset_name == 'LSMDC':
            if split_type == 'train':
                dataset = LSMDCDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                dataset = LSMDCDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == 'ACTNET':
            if split_type == 'train':
                dataset = ActivityNetDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                dataset = ActivityNetDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == 'DIDEMO':
            if split_type == 'train':
                dataset = DiDeMoDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                dataset = DiDeMoDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

        else:
            raise NotImplementedError

    @staticmethod
    def get_train_video_ids(config: Config):
        """
        Get unique training video IDs for expanded pool evaluation.

        Returns:
            tuple: (video_ids, videos_dir, video_ext, path_fn)
                - video_ids: List of unique video IDs
                - videos_dir: Directory containing videos
                - video_ext: Video file extension
                - path_fn: Optional function to build video paths (for complex paths)
        """
        if config.dataset_name == "MSRVTT":
            if config.msrvtt_train_file == '7k':
                train_csv = 'reranker/xpool/data/MSRVTT/MSRVTT_train.7k.csv'
            else:
                train_csv = 'reranker/xpool/data/MSRVTT/MSRVTT_train.9k.csv'
            train_df = pd.read_csv(train_csv)
            video_ids = train_df['video_id'].unique().tolist()
            return video_ids, config.videos_dir, '.mp4', None

        elif config.dataset_name == "MSVD":
            train_file = 'data/MSVD/train_list.txt'
            video_ids = read_lines(train_file)
            return video_ids, config.videos_dir, '.avi', None

        elif config.dataset_name == "LSMDC":
            train_file = 'reranker/xpool/data/LSMDC/LSMDC16_annos_training.csv'
            clip_ids = []
            with open(train_file, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    line_split = line.split("\t")
                    if len(line_split) == 6:
                        clip_id = line_split[0]
                        if clip_id != '1012_Unbreakable_00.05.16.065-00.05.21.941':
                            clip_ids.append(clip_id)

            def lsmdc_path_fn(videos_dir, clip_id, video_ext):
                clip_prefix = clip_id.split('.')[0][:-3]
                return os.path.join(videos_dir, clip_prefix, clip_id + video_ext)

            return clip_ids, config.videos_dir, '.avi', lsmdc_path_fn

        elif config.dataset_name == "ACTNET":
            anno_file = 'reranker/xpool/data/ACTNET/actnet_ret_train.json'
            annotations = load_json(anno_file)
            # Strip .mp4 suffix to match cache file naming
            video_ids = [item['video'].replace('.mp4', '') for item in annotations]

            def actnet_path_fn(videos_dir, vid, video_ext):
                # vid does not include .mp4, add it back
                return os.path.join(videos_dir, vid + '.mp4')

            return video_ids, config.videos_dir, '', actnet_path_fn

        elif config.dataset_name == "DIDEMO":
            anno_file = 'reranker/xpool/data/DIDEMO/didemo_ret_train.json'
            annotations = load_json(anno_file)
            # Strip .mp4 suffix to match cache file naming
            video_ids = [item['video'].replace('.mp4', '') for item in annotations]
            videos_dir = os.path.join(config.videos_dir, 'train', 'videos')

            def didemo_path_fn(videos_dir, vid, video_ext):
                # vid does not include .mp4, add it back
                return os.path.join(videos_dir, vid + '.mp4')

            return video_ids, videos_dir, '', didemo_path_fn

        else:
            raise NotImplementedError(f"Dataset {config.dataset_name} not supported for expanded pool")

    @staticmethod
    def get_video_only_loader(config: Config, video_ids: list, videos_dir: str,
                               video_ext: str = '.mp4', path_fn=None):
        """
        Create DataLoader for video-only loading (no text).

        Used for loading additional videos to expand the search pool.

        Args:
            config: Configuration object
            video_ids: List of video IDs to load
            videos_dir: Directory containing videos
            video_ext: Video file extension
            path_fn: Optional function to build video paths

        Returns:
            DataLoader for video-only dataset
        """
        img_transforms = init_transform_dict(config.input_res)['clip_test']
        dataset = VideoOnlyDataset(
            config, video_ids, videos_dir, img_transforms, video_ext, path_fn
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
