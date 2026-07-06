import os
import re
import cv2
import random
import numpy as np
import torch


class VideoCapture:
    FRAME_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    @staticmethod
    def _sample_frame_indices(num_items, num_frames, sample='rand'):
        if num_items <= 0:
            raise ValueError("Cannot sample frames from an empty source")

        acc_samples = min(num_frames, num_items)
        intervals = np.linspace(start=0, stop=num_items, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, start in enumerate(intervals[:-1]):
            end = max(start, intervals[idx + 1] - 1)
            ranges.append((start, end))

        if sample == 'rand':
            return [random.randint(start, end) for start, end in ranges]
        return [(start + end) // 2 for start, end in ranges]

    @staticmethod
    def _finalize_frames(frames, frame_idxs, num_frames):
        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
        frames = torch.stack(frames).float() / 255
        return frames, frame_idxs

    @staticmethod
    def _frame_sort_key(frame_path):
        name = os.path.basename(frame_path)
        return (tuple(int(part) for part in re.findall(r'\d+', name)), name.lower())

    @staticmethod
    def load_frames(media_path, num_frames, sample='rand'):
        if os.path.isdir(media_path):
            return VideoCapture.load_frames_from_directory(media_path, num_frames, sample)
        return VideoCapture.load_frames_from_video(media_path, num_frames, sample)

    @staticmethod
    def load_frames_from_directory(frame_dir, num_frames, sample='rand'):
        frame_paths = []
        for name in os.listdir(frame_dir):
            if name.startswith('._'):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in VideoCapture.FRAME_EXTENSIONS:
                frame_paths.append(os.path.join(frame_dir, name))

        frame_paths.sort(key=VideoCapture._frame_sort_key)
        if not frame_paths:
            raise FileNotFoundError(f"No frame images found in: {frame_dir}")

        frame_idxs = VideoCapture._sample_frame_indices(len(frame_paths), num_frames, sample)
        frames = []
        for index in frame_idxs:
            frame = cv2.imread(frame_paths[index], cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to read frame: {frame_paths[index]}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(frame)

        return VideoCapture._finalize_frames(frames, frame_idxs, num_frames)

    @staticmethod
    def load_frames_from_video(video_path, num_frames, sample='rand'):
        """
            video_path: str/os.path
            num_frames: int - number of frames to sample
            sample: 'rand' | 'uniform' how to sample
            returns: frames: torch.tensor of stacked sampled video frames
                             of dim (num_frames, C, H, W)
                     idxs: list(int) indices of where the frames where sampled
        """
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), video_path
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = VideoCapture._sample_frame_indices(vlen, num_frames, sample)

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                raise ValueError

        cap.release()
        return VideoCapture._finalize_frames(frames, frame_idxs, num_frames)
