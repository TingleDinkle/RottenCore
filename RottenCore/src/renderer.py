import cv2
import numpy as np
import torch
import os # For checking file existence
from src.compression import RottenCompressor

class RottenRenderer:
    def __init__(self, rc_file_path: str):
        if not os.path.exists(rc_file_path):
            raise FileNotFoundError(f"RottenCore project file not found: {rc_file_path}")

        blocks, block_sequence, metadata = RottenCompressor.load_project(rc_file_path)
        
        self.blocks = blocks # Expects a torch.Tensor
        self.width = metadata['width']
        self.height = metadata['height']
        self.block_size = metadata['block_size']
        self.num_glyphs = metadata['num_glyphs']
        self.block_sequence = block_sequence # List of numpy arrays, one per frame
        self.original_fps = metadata.get('original_fps', 30.0) # Load original_fps, default to 30.0

        self.block_h, self.block_w = self.block_size
        self.num_blocks_x = self.width // self.block_w
        self.num_blocks_y = self.height // self.block_h

        # Move blocks to device for efficient access
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Blocks might be (N, 1, H, W) from ML path, or (N, H, W) from K-Means path.
        # Ensure it's (N, H, W) for consistency.
        if self.blocks.dim() == 4 and self.blocks.shape[1] == 1:
            self.blocks_tensor = self.blocks.to(self.device).squeeze(1) 
        else:
            self.blocks_tensor = self.blocks.to(self.device)


    def _build_frame_from_glyphs(self, glyph_indices: np.ndarray) -> np.ndarray:
        """
        Reconstructs a grayscale image frame from a sequence of glyph indices.
        glyph_indices: 1D numpy array of glyph IDs for one frame.
        Returns: numpy array (H, W) in [0, 255], grayscale.
        """
        reconstructed_frame_tensor = torch.zeros((self.height, self.width), device=self.device, dtype=torch.float32)
        
        block_idx_in_sequence = 0
        for y_block in range(self.num_blocks_y):
            for x_block in range(self.num_blocks_x):
                glyph_id = glyph_indices[block_idx_in_sequence]
                block_data = self.blocks_tensor[glyph_id] # (block_h, block_w)
                
                reconstructed_frame_tensor[y_block * self.block_h : (y_block + 1) * self.block_h,
                                           x_block * self.block_w : (x_block + 1) * self.block_w] = block_data
                block_idx_in_sequence += 1
        
        # Convert to numpy, scale to 0-255, and return
        return (reconstructed_frame_tensor.cpu().numpy() * 255).astype(np.uint8)

    def render_video(self, output_filepath: str, scale: int = 1, fps: float = None):
        """
        Renders a video from the loaded glyph sequence.
        
        Args:
            output_filepath (str): Path to save the output video (e.g., output.mp4).
            scale (int): Upscaling factor for the output video.
            fps (float): Frames per second for the output video. If None, uses original_fps from project data.
        """
        if not self.block_sequence:
            print("Warning: No block sequence found in project data. Cannot render video.")
            return

        final_width = self.width * scale
        final_height = self.height * scale
        
        # Use provided FPS or fallback to original_fps
        fps_to_use = fps if fps is not None else self.original_fps
        if fps_to_use is None: # Final fallback
            fps_to_use = 30.0
            print("Warning: FPS not specified and not found in project data. Defaulting to 30.0 FPS.")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
        out = cv2.VideoWriter(output_filepath, fourcc, float(fps_to_use), (final_width, final_height), isColor=False)

        if not out.isOpened():
            raise IOError(f"Could not open video writer for {output_filepath}")

        print(f"Rendering video to {output_filepath} (Resolution: {final_width}x{final_height}, FPS: {fps_to_use})...")
        for frame_idx, glyph_indices_frame in enumerate(tqdm(self.block_sequence, desc="Reconstructing frames")):
            # Build the base resolution frame
            base_frame = self._build_frame_from_glyphs(glyph_indices_frame)

            # Apply scaling if necessary
            if scale > 1:
                scaled_frame = cv2.resize(base_frame, (final_width, final_height), interpolation=cv2.INTER_NEAREST)
                out.write(scaled_frame)
            else:
                out.write(base_frame)
        
        out.release()
        print(f"Rendered video saved to {output_filepath}")