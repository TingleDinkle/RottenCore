import torch
import torch.nn as nn
import torchvision.utils
import lpips
from torch.utils.data import DataLoader
import math
import os
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

from .video_utils import VideoFramesDataset
from .compression import RottenCompressor
from .extreme_compression import ExtremeCompressor

class ImageReconstruction(nn.Module):
    def __init__(self, width: int, height: int, block_size: tuple = (8, 8), n_blocks: int = 256):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.img_size = (height, width) # img_size is (height, width)

        self.tiles_per_img = (self.img_size[0] * self.img_size[1]) // (self.block_size[0] * self.block_size[1])

        self.unfold = torch.nn.Unfold(kernel_size=self.block_size, stride=self.block_size)
        self.fold = torch.nn.Fold(output_size=self.img_size, kernel_size=self.block_size, stride=self.block_size)

        self.blocks = None
        self.invert_blocks = False # Keeping this for potential future use or to match original intent, but not actively used for inversion in current forward pass

        self.init_blocks_haar() # Initialize blocks using Haar-like features by default

    def init_blocks_haar(self):
        """
        Initialize blocks as something vaguely resembling Haar cascades.
        Good for getting an orthogonal-ish initial set.
        Adapted from original train_patches.py
        """
        # A simpler set of basis for initial Haar-like blocks
        basis_1d = np.array([
            [0,0,0,0,0,0,0,0], [0,0,0,0,1,1,1,1], [0,0,1,1,1,1,0,0], [0,1,1,1,1,0,0,0],
            [1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,1,0,0,0,0,0], [1,1,1,1,0,0,0,0]
        ], dtype=np.bool_)

        out_blocks = []
        out_tuple = set()

        block_h, block_w = self.block_size

        # Generate 2D blocks from 1D basis
        # This is a simplified version of the original generation
        # The original Haar generation was quite complex and produced many candidates.
        # We'll aim for a diverse set up to n_blocks.
        for i in range(basis_1d.shape[0]):
            for j in range(basis_1d.shape[0]):
                block_candidate = np.outer(basis_1d[i][:block_h], basis_1d[j][:block_w])

                # Variations: original, inverted, transposed, inverted-transposed
                candidates = [
                    block_candidate,
                    np.logical_not(block_candidate),
                    block_candidate.T,
                    np.logical_not(block_candidate.T)
                ]

                for c in candidates:
                    if len(out_blocks) >= self.n_blocks:
                        break
                    c_tuple = tuple(map(tuple, c))
                    if c_tuple not in out_tuple:
                        out_tuple.add(c_tuple)
                        out_blocks.append(c)
            if len(out_blocks) >= self.n_blocks:
                break

        # If not enough blocks from Haar, fill with random or simple patterns
        while len(out_blocks) < self.n_blocks:
            random_block = np.random.rand(block_h, block_w) > 0.5
            r_tuple = tuple(map(tuple, random_block))
            if r_tuple not in out_tuple:
                out_tuple.add(r_tuple)
                out_blocks.append(random_block)


        out_blocks_np = np.array(out_blocks, dtype=np.float32)

        # blur these a little to make gradients more likely
        # Only if block_size is large enough for gaussian_filter to make sense
        if min(self.block_size) > 1:
            for i in range(len(out_blocks_np)):
                out_blocks_np[i] = gaussian_filter(out_blocks_np[i], sigma=np.sqrt(2))


        self.blocks = nn.Parameter(torch.Tensor(out_blocks_np), requires_grad=True)

    def forward(self, target_img: torch.Tensor):
        beta = 35 # Sharpening factor applied to block matches. 

        block_len = math.prod(self.block_size)

        # target_img shape: (batch_size, 1, H, W) -> unfold -> (batch_size, block_len, num_patches)
        uf_tgt = self.unfold(target_img)

        n_effective_blocks = self.n_blocks
        block_cd = self.blocks[:n_effective_blocks] # Use only up to n_blocks

        # Reshape for comparison:
        # a: (batch_size * num_patches, 1, block_len)
        # b: (1, n_effective_blocks, block_len, 1) -> (1, n_effective_blocks, block_len)
        a = uf_tgt.permute(0, 2, 1).reshape(-1, 1, block_len) # (batch_size * num_patches, 1, block_len)
        b = block_cd.view(1, n_effective_blocks, block_len)  # (1, n_effective_blocks, block_len)

        sqerr = torch.sum(torch.square(a - b), dim=2)

        # (sharpened) softmax over block-choice dim
        block_weights = torch.softmax(-1 * sqerr * beta, dim=1) # (batch_size * num_patches, n_effective_blocks)

        # Reconstruct unfolded image by weighted sum of blocks
        # uf_reconstructed shape: (batch_size * num_patches, block_len)
        uf_reconstructed = torch.matmul(block_weights, block_cd.view(n_effective_blocks, block_len))

        # Reshape back to (batch_size, block_len, num_patches) for fold
        uf_reconstructed = uf_reconstructed.view(target_img.shape[0], self.tiles_per_img, block_len).permute(0, 2, 1)

        # Fold back into expected image shape (batch_size, 1, H, W)
        reconstructed = self.fold(uf_reconstructed)

        choice_penalty = torch.mean(torch.minimum(block_weights, 1 - block_weights)) * n_effective_blocks

        return reconstructed, choice_penalty

    def get_best_block_indices(self, target_img: torch.Tensor):
        """
        For a given batch of target images, return the index of the best matching block for each patch.
        target_img shape: (batch_size, 1, H, W)
        Returns: (batch_size, num_patches) tensor of indices
        """
        block_len = math.prod(self.block_size)
        uf_tgt = self.unfold(target_img) # (batch_size, block_len, num_patches)

        n_effective_blocks = self.n_blocks
        block_cd = self.blocks[:n_effective_blocks]

        a = uf_tgt.permute(0, 2, 1).reshape(-1, 1, block_len) # (batch_size * num_patches, 1, block_len)
        b = block_cd.view(1, n_effective_blocks, block_len)  # (1, n_effective_blocks, block_len)

        sqerr = torch.sum(torch.square(a - b), dim=2) # (batch_size * num_patches, n_effective_blocks)

        # Find the index of the minimum squared error for each patch
        _, min_indices = torch.min(sqerr, dim=1)

        return min_indices.view(target_img.shape[0], self.tiles_per_img)

def train_glyphs(
    video_path: str,
    output_rc_path: str,
    width: int,
    height: int,
    num_glyphs: int = 256,
    block_size: tuple = (8, 8),
    epochs: int = 500, # Increased epochs for better training
    learning_rate: float = 0.002,
    batch_size: int = 48,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    original_fps: float = 30.0 # Added original_fps
):
    print(f"Training glyphs for video: {video_path}")
    print(f"Target resolution: {width}x{height}, Glyphs: {num_glyphs}, Block size: {block_size}")
    print(f"Device: {device}")

    # Set up device
    actual_device = torch.device(device)

    # Load video data
    dataset = VideoFramesDataset(video_path, width, height, actual_device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Image Reconstruction model
    recr_model = ImageReconstruction(width, height, block_size, num_glyphs).to(actual_device)

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(recr_model.parameters(), lr=learning_rate)
    perceptual_loss_fn = lpips.LPIPS(net='alex').to(actual_device) # Using 'alex' net for LPIPS

    print("Starting training...")
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        epoch_loss_p = 0.0
        epoch_loss_c = 0.0
        num_batches = 0

        for target_img_batch, _ in data_loader:
            optimizer.zero_grad()

            reconstructed_img_batch, choice_penalty = recr_model(target_img_batch)

            # LPIPS expects 3 color channels, convert grayscale to 3 channels
            target_img_rgb = target_img_batch.repeat(1, 3, 1, 1)
            reconstructed_img_rgb = reconstructed_img_batch.repeat(1, 3, 1, 1)

            # Upsample for LPIPS
            # LPIPS usually expects larger images, typically 256x256 or similar,
            # so upsampling to a reasonable size is good practice if original images are small.
            # Using original logic of doubling size for LPIPS input if small.
            lpips_input_size = (recr_model.img_size[0] * 2, recr_model.img_size[1] * 2)
            if recr_model.img_size[0] < 256 or recr_model.img_size[1] < 256:
                 target_img_rgb = nn.functional.interpolate(target_img_rgb, size=lpips_input_size, mode='bilinear', align_corners=False)
                 reconstructed_img_rgb = nn.functional.interpolate(reconstructed_img_rgb, size=lpips_input_size, mode='bilinear', align_corners=False)


            loss_p = perceptual_loss_fn(reconstructed_img_rgb, target_img_rgb).mean()
            loss_c = choice_penalty

            loss = loss_p + loss_c

            loss.backward()
            optimizer.step()

            # Clamp parameter range for blocks
            for p in recr_model.parameters():
                if p.grad is not None:
                    p.data.clamp_(0.0, 1.0)

            epoch_loss += loss.item()
            epoch_loss_p += loss_p.item()
            epoch_loss_c += loss_c.item()
            num_batches += 1

        if num_batches > 0:
            epoch_loss /= num_batches
            epoch_loss_p /= num_batches
            epoch_loss_c /= num_batches

        # tqdm.set_postfix doesn't work well inside DataLoader loop, so print after epoch
        if epoch % 10 == 0 or epoch == epochs - 1: # Print every 10 epochs or last epoch
            tqdm.write(f"Epoch {epoch+1}/{epochs}: Total Loss: {epoch_loss:.6f}, Perceptual Loss: {epoch_loss_p:.6f}, Choice Penalty: {epoch_loss_c:.6f}")


    print("\nTraining complete. Generating block sequence...")
    # Generate block sequence for the entire video using the trained glyphs
    recr_model.eval() # Set model to evaluation mode
    block_sequence = []
    with torch.no_grad(): # Disable gradient calculations
        for frame_tensor, _ in tqdm(dataset, desc="Generating sequence"):
            # frame_tensor is (1, H, W). Need to add batch dim: (1, 1, H, W)
            frame_tensor = frame_tensor.unsqueeze(0).to(actual_device)
            glyph_indices_frame = recr_model.get_best_block_indices(frame_tensor)
            block_sequence.append(glyph_indices_frame.squeeze(0).cpu().numpy()) # Remove batch dim, to numpy

    print("Saving project file...")
    
    # Save the trained blocks and metadata
    # Note: RottenCompressor.save_project might be monkey-patched to ExtremeCompressor.save_project
    # by the calling script (rottencore.py) if extreme mode is enabled.
    RottenCompressor.save_project(
        output_rc_path,
        blocks=recr_model.blocks.detach().cpu(),
        block_sequence=block_sequence,
        metadata={
            'width': width,
            'height': height,
            'block_size': block_size,
            'num_glyphs': num_glyphs,
            'original_fps': original_fps
        }
    )
    print(f"Project saved to {output_rc_path}")
