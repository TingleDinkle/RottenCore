import cv2
import numpy as np
import torch
import os
import imageio
from tqdm import tqdm
from .compression import RottenCompressor

class RottenRenderer:
    def __init__(self, rc_file_path: str):
        if not os.path.exists(rc_file_path):
            raise FileNotFoundError(f"RottenCore project file not found: {rc_file_path}")

        blocks, block_sequence, metadata = RottenCompressor.load_project(rc_file_path)
        
        self.blocks = blocks
        self.width = metadata['width']
        self.height = metadata['height']
        self.block_size = metadata['block_size']
        self.num_glyphs = metadata['num_glyphs']
        # block_sequence comes pre-shaped from the new compression loader
        self.block_sequence = block_sequence 
        self.original_fps = metadata.get('original_fps', 30.0)

        self.block_h, self.block_w = self.block_size
        self.num_blocks_x = self.width // self.block_w
        self.num_blocks_y = self.height // self.block_h

        # Move blocks to device for efficient access
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.blocks.dim() == 4 and self.blocks.shape[1] == 1:
            self.blocks_tensor = self.blocks.to(self.device).squeeze(1) 
        else:
            self.blocks_tensor = self.blocks.to(self.device)


    def _build_frame_from_glyphs(self, glyph_indices: np.ndarray) -> np.ndarray:
        reconstructed_frame_tensor = torch.zeros((self.height, self.width), device=self.device, dtype=torch.float32)
        
        block_idx_in_sequence = 0
        # glyph_indices is just 1D array for this frame
        for y_block in range(self.num_blocks_y):
            for x_block in range(self.num_blocks_x):
                glyph_id = glyph_indices[block_idx_in_sequence]
                block_data = self.blocks_tensor[glyph_id]
                
                reconstructed_frame_tensor[y_block * self.block_h : (y_block + 1) * self.block_h,
                                           x_block * self.block_w : (x_block + 1) * self.block_w] = block_data
                block_idx_in_sequence += 1
        
        return (reconstructed_frame_tensor.cpu().numpy() * 255).astype(np.uint8)

    def render_video(self, output_filepath: str, scale: int = 1, fps: float = None):
        if self.block_sequence is None or len(self.block_sequence) == 0:
            print("Warning: No block sequence found in project data. Cannot render video.")
            return

        final_width = self.width * scale
        final_height = self.height * scale
        
        fps_to_use = fps if fps is not None else self.original_fps
        if fps_to_use is None:
            fps_to_use = 30.0
            print("Warning: FPS not specified. Defaulting to 30.0 FPS.")

        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Render frames
        rendered_frames = []
        print(f"Reconstructing frames for {output_filepath}...")
        
        for glyph_indices_frame in tqdm(self.block_sequence, desc="Reconstructing"):
            base_frame = self._build_frame_from_glyphs(glyph_indices_frame)
            if scale > 1:
                scaled_frame = cv2.resize(base_frame, (final_width, final_height), interpolation=cv2.INTER_NEAREST)
                rendered_frames.append(scaled_frame)
            else:
                rendered_frames.append(base_frame)

        # Write output based on extension
        ext = os.path.splitext(output_filepath)[1].lower()

        if ext == '.gif':
            print(f"Saving GIF to {output_filepath}...")
            # imageio expects [H, W, C] or [H, W]
            # Since frames are grayscale, they are [H, W]. imageio handles this.
            imageio.mimsave(output_filepath, rendered_frames, fps=fps_to_use, loop=0)
        else:
            print(f"Saving video to {output_filepath}...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_filepath, fourcc, float(fps_to_use), (final_width, final_height), isColor=False)
            if not out.isOpened():
                raise IOError(f"Could not open video writer for {output_filepath}")
            
            for frame in rendered_frames:
                out.write(frame)
            out.release()

        print(f"Saved to {output_filepath}")
