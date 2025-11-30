import torch
import numpy as np
import pickle
import os

# No longer needing cv2, tqdm, VideoFramesDataset directly in RottenCoreProject
# from .video_utils import VideoFramesDataset

class RottenCoreProject:
    """
    Handles loading and providing access to the RottenCore project data.
    """
    def __init__(self, rc_file_path: str):
        if not os.path.exists(rc_file_path):
            raise FileNotFoundError(f"RottenCore project file not found: {rc_file_path}")

        with open(rc_file_path, 'rb') as f:
            project_data = pickle.load(f)
        
        self.blocks = project_data['blocks'] # Expects a torch.Tensor
        self.width = project_data['width']
        self.height = project_data['height']
        self.block_size = project_data['block_size']
        self.num_glyphs = project_data['num_glyphs']
        # The block_sequence is now part of the project file.
        # It's a list of numpy arrays, where each numpy array represents the glyph IDs for one frame.
        self.block_sequence = project_data['block_sequence']