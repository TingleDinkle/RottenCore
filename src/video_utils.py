import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

def get_video_properties(video_path):
    """
    Returns the width, height, and frames per second (fps) of a video.

    :param video_path: Path to the video file.
    :return: Tuple (width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

class VideoFramesDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing video frames.
    """
    def __init__(self, video_path, target_width, target_height):
        self.video_path = video_path
        self.target_width = target_width
        self.target_height = target_height
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames == 0:
            raise ValueError(f"Video file {video_path} contains no frames.")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path) # Reopen if closed
            if not self.cap.isOpened():
                raise IOError(f"Cannot reopen video file: {self.video_path}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()

        if not ret:
            # Try to read again or handle corrupted frame
            print(f"Warning: Could not read frame {idx} from {self.video_path}. Retrying...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from {self.video_path}")

        # Resize frame
        frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)

        # Convert to Grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize to [0, 1] range
        normalized_frame = gray_frame.astype(np.float32) / 255.0

        # Return as a Tensor (1, H, W)
        # Add a channel dimension for grayscale (1 channel)
        tensor_frame = torch.from_numpy(normalized_frame).unsqueeze(0)

        return tensor_frame

    def close(self):
        """Releases the video capture object."""
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.close()

