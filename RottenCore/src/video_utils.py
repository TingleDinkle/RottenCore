import av
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoFramesDataset(Dataset):
    """
    Loads a video, downsamples it, and caches all the frames on the specified device.
    """
    def __init__(self, video_path: str, width: int, height: int, device: torch.device):
        self.device = device
        self.frames = []
        self.width = width
        self.height = height
        self._load_data(video_path)

    def _load_data(self, video_path: str):
        container = av.open(video_path)
        for frame in container.decode(video=0):
            frame_np = frame.to_ndarray(format='bgr24')
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
            frame_np = cv2.resize(frame_np, (self.width, self.height), interpolation=cv2.INTER_AREA)
            frame_np = frame_np.astype(np.float32)[None, ...] / 255.0
            frame_np = torch.from_numpy(frame_np).to(self.device)
            self.frames.append(frame_np)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        return self.frames[idx], idx

def get_video_properties(video_path: str):
    """
    Retrieves the original width, height, and frame rate of a video.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    return stream.width, stream.height, stream.average_rate
