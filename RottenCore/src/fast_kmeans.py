import numpy as np
from numba import njit, prange
from tqdm import tqdm
import math
import torch # For converting to/from VideoFramesDataset
import pickle # For saving project data

from .video_utils import VideoFramesDataset # Import to use VideoFramesDataset

# Constants from common/bacommon.h
BLOCKSIZE = 8
KMEANS_INITIAL_CENTROIDS = 2048
KMEANS_ITERATIONS = 250
GLYPH_COUNT_REDUCE_PER_FRAME = 8 # Not directly used for iterations, but for reducing centroids
GLYPH_INVERSION_MASK = 0x800 # From common/bacommon.h, used in ComputeDistance

@njit(parallel=True)
def _compute_distance_numba(block_intensity, centroid_intensity):
    """
    Computes the squared Euclidean distance between a block and a centroid.
    Handles optional inversion as per C code.
    block_intensity: (BLOCKSIZE*BLOCKSIZE,) numpy array
    centroid_intensity: (BLOCKSIZE*BLOCKSIZE,) numpy array
    Returns: (distance, inversion_applied_flag)
    """
    diff = 0.0
    for i in prange(block_intensity.shape[0]):
        d = block_intensity[i] - centroid_intensity[i]
        diff += d * d # MSE

    # Check for inversion (as per C code's ALLOW_GLYPH_INVERSION logic)
    inv_diff = 0.0
    for i in prange(block_intensity.shape[0]):
        d_inv = block_intensity[i] - (1.0 - centroid_intensity[i])
        inv_diff += d_inv * d_inv
    
    if inv_diff < diff:
        return inv_diff, GLYPH_INVERSION_MASK
    else:
        return diff, 0 # No inversion applied


@njit(parallel=True)
def _assign_to_centroids_numba(blocks, centroids, assignments, distances, inversion_flags):
    """
    Assigns each block to the closest centroid.
    blocks: (num_blocks, BLOCKSIZE*BLOCKSIZE) numpy array
    centroids: (num_centroids, BLOCKSIZE*BLOCKSIZE) numpy array
    assignments: (num_blocks,) output array for centroid indices
    distances: (num_blocks,) output array for min distances
    inversion_flags: (num_blocks,) output array for inversion status
    """
    num_blocks = blocks.shape[0]
    num_centroids = centroids.shape[0]

    for i in prange(num_blocks):
        min_dist = np.finfo(np.float32).max
        best_centroid_idx = -1
        best_inversion_flag = 0

        for j in range(num_centroids):
            current_dist, current_inversion_flag = _compute_distance_numba(blocks[i], centroids[j])
            
            if current_dist < min_dist:
                min_dist = current_dist
                best_centroid_idx = j
                best_inversion_flag = current_inversion_flag
        
        assignments[i] = best_centroid_idx
        distances[i] = min_dist
        inversion_flags[i] = best_inversion_flag

@njit(parallel=True)
def _update_centroids_numba(blocks, assignments, inversion_flags, initial_centroid_indices, new_centroids_sum_accumulator, new_counts_accumulator):
    """
    Updates centroids based on assignments.
    blocks: (num_blocks, BLOCKSIZE*BLOCKSIZE) numpy array (flattened)
    assignments: (num_blocks,) numpy array of assigned centroid indices (from the active set)
    inversion_flags: (num_blocks,) numpy array of inversion flags for each assignment
    initial_centroid_indices: (num_blocks,) numpy array, original index of the assigned centroid (before filtering dead)
    new_centroids_sum_accumulator: (num_initial_centroids, BLOCKSIZE*BLOCKSIZE) output array for sum of intensities
    new_counts_accumulator: (num_initial_centroids,) output array for counts
    """
    num_initial_centroids = new_centroids_sum_accumulator.shape[0]
    # new_centroids_sum_accumulator[:] = 0.0 # Reset sums - handled outside for clarity
    # new_counts_accumulator[:] = 0      # Reset counts - handled outside for clarity

    for i in prange(blocks.shape[0]):
        assigned_original_idx = initial_centroid_indices[i]
        
        if assigned_original_idx == -1:
            continue

        new_counts_accumulator[assigned_original_idx] += 1
        
        if inversion_flags[i] == GLYPH_INVERSION_MASK:
            for k in prange(blocks.shape[1]):
                new_centroids_sum_accumulator[assigned_original_idx, k] += (1.0 - blocks[i, k])
        else:
            for k in prange(blocks.shape[1]):
                new_centroids_sum_accumulator[assigned_original_idx, k] += blocks[i, k]


@njit(parallel=True)
def _find_best_glyph_for_patches(patches, final_glyphs_flat):
    """
    Finds the best matching glyph for each patch.
    patches: (num_patches_in_frame, block_len)
    final_glyphs_flat: (num_final_glyphs, block_len)
    Returns: (num_patches_in_frame,) numpy array of best glyph indices
    """
    num_patches = patches.shape[0]
    num_glyphs = final_glyphs_flat.shape[0]
    best_glyph_indices = np.empty(num_patches, dtype=np.int32)

    for i in prange(num_patches):
        min_dist = np.finfo(np.float32).max
        best_idx = -1
        for j in range(num_glyphs):
            # No inversion check here, as final glyphs are fixed.
            # We don't care about the inversion status for final assignment, just raw distance
            d_sq = 0.0
            for k in prange(patches.shape[1]):
                d = patches[i, k] - final_glyphs_flat[j, k]
                d_sq += d * d
            
            if d_sq < min_dist:
                min_dist = d_sq
                best_idx = j
        best_glyph_indices[i] = best_idx
    return best_glyph_indices


def generate_glyphs_kmeans(video_path: str, output_rc_path: str, width: int, height: int, num_glyphs: int = 256, block_size: tuple = (BLOCKSIZE, BLOCKSIZE), original_fps: float = 30.0):
    """
    Generates glyphs using a K-Means-like algorithm inspired by the C implementation.
    
    Args:
        video_path (str): Path to the input video file.
        output_rc_path (str): Path to save the RottenCore project file.
        width (int): Target width for video frames.
        height (int): Target height for video frames.
        num_glyphs (int): The target number of glyphs (K in K-Means).
        block_size (tuple): The (height, width) of each block.
        original_fps (float): Original frames per second of the input video.

    Returns:
        None. Saves the project data to output_rc_path.
    """
    block_h, block_w = block_size
    block_len = block_h * block_w

    if block_h != BLOCKSIZE or block_w != BLOCKSIZE:
        print(f"Warning: BLOCKSIZE in C code was {BLOCKSIZE}x{BLOCKSIZE}, but block_size is {block_size}. This implementation assumes {BLOCKSIZE}x{BLOCKSIZE} for direct C porting comparison.")

    # Load video data once for patch extraction
    # Using cpu for video_dataset as Numba will operate on numpy arrays anyway.
    temp_device = torch.device("cpu")
    video_dataset = VideoFramesDataset(video_path, width, height, temp_device)

    all_patches_list = []
    frames_shape = [] # Store original frame shapes to reconstruct block_sequence
    
    print("Extracting patches from video frames...")
    for frame_idx, (frame_tensor, _) in enumerate(tqdm(video_dataset, desc="Extracting patches")):
        # frame_tensor is (1, H, W)
        frame_np = frame_tensor.squeeze(0).cpu().numpy() # (H, W)
        frames_shape.append(frame_np.shape) # Store (H,W) of processed frame
        
        num_blocks_h = frame_np.shape[0] // block_h
        num_blocks_w = frame_np.shape[1] // block_w

        for y in range(num_blocks_h):
            for x in range(num_blocks_w):
                patch = frame_np[y*block_h:(y+1)*block_h, x*block_w:(x+1)*block_w]
                all_patches_list.append(patch.flatten()) # Flatten for easier processing

    all_patches = np.array(all_patches_list, dtype=np.float32) # (num_total_patches, block_len)
    num_total_patches = all_patches.shape[0]

    print(f"Total {num_total_patches} patches extracted across {len(video_dataset)} frames.")

    # Initialize centroids randomly
    centroids = np.random.rand(KMEANS_INITIAL_CENTROIDS, block_len).astype(np.float32)

    centroid_is_dead = np.zeros(KMEANS_INITIAL_CENTROIDS, dtype=np.bool_)
    num_active_centroids = KMEANS_INITIAL_CENTROIDS

    patch_assignments = np.full(num_total_patches, -1, dtype=np.int32)
    patch_original_centroid_indices = np.full(num_total_patches, -1, dtype=np.int32)
    patch_min_distances = np.zeros(num_total_patches, dtype=np.float32)
    patch_inversion_flags = np.zeros(num_total_patches, dtype=np.int32)

    # For centroid updates
    new_centroids_sum_accumulator = np.zeros_like(centroids, dtype=np.float32)
    new_counts_accumulator = np.zeros(KMEANS_INITIAL_CENTROIDS, dtype=np.int32)


    print("Starting K-Means iterations...")
    for iteration in tqdm(range(KMEANS_ITERATIONS), desc="K-Means Iterations"):
        active_centroid_indices = np.where(~centroid_is_dead)[0]
        active_centroids = centroids[active_centroid_indices]
        
        if len(active_centroids) == 0:
            print("No active centroids left. Stopping K-Means.")
            break

        _assign_to_centroids_numba(all_patches, active_centroids, patch_assignments, patch_min_distances, patch_inversion_flags)
        
        for i in range(num_total_patches):
            if patch_assignments[i] != -1:
                patch_original_centroid_indices[i] = active_centroid_indices[patch_assignments[i]]
            else:
                patch_original_centroid_indices[i] = -1

        new_centroids_sum_accumulator[:] = 0.0 # Reset sums for current iteration
        new_counts_accumulator[:] = 0      # Reset counts for current iteration
        _update_centroids_numba(all_patches, patch_assignments, patch_inversion_flags, 
                                patch_original_centroid_indices, new_centroids_sum_accumulator, new_counts_accumulator)
        
        for c_idx in range(KMEANS_INITIAL_CENTROIDS):
            if not centroid_is_dead[c_idx] and new_counts_accumulator[c_idx] > 0:
                centroids[c_idx] = new_centroids_sum_accumulator[c_idx] / new_counts_accumulator[c_idx]
            elif not centroid_is_dead[c_idx] and new_counts_accumulator[c_idx] == 0:
                centroid_is_dead[c_idx] = True
                num_active_centroids -= 1
        
        if num_active_centroids > num_glyphs:
            active_counts = new_counts_accumulator[~centroid_is_dead]
            active_indices = np.where(~centroid_is_dead)[0]

            if len(active_indices) > 0:
                sorted_by_count_indices = active_indices[np.argsort(active_counts)]
                
                num_to_kill = min(len(sorted_by_count_indices), num_active_centroids - num_glyphs)
                num_to_kill = min(num_to_kill, GLYPH_COUNT_REDUCE_PER_FRAME)

                for i in range(num_to_kill):
                    centroid_to_kill_original_idx = sorted_by_count_indices[i]
                    if not centroid_is_dead[centroid_to_kill_original_idx]:
                        centroid_is_dead[centroid_to_kill_original_idx] = True
                        num_active_centroids -= 1
        
        if num_active_centroids <= num_glyphs:
            final_glyphs_list = []
            final_count = 0
            for c_idx in range(KMEANS_INITIAL_CENTROIDS):
                if not centroid_is_dead[c_idx]:
                    final_glyphs_list.append(centroids[c_idx])
                    final_count += 1
                    if final_count == num_glyphs:
                        break
            
            if final_count == num_glyphs:
                final_glyphs = np.array(final_glyphs_list)
                print(f"K-Means converged to {num_glyphs} glyphs.")
                break # Exit K-Means loop early if target reached

    print(f"\nK-Means iterations finished. Finalizing to {num_glyphs} glyphs.")
    final_glyphs_list = []
    for c_idx in range(KMEANS_INITIAL_CENTROIDS):
        if not centroid_is_dead[c_idx]:
            final_glyphs_list.append(centroids[c_idx])
        if len(final_glyphs_list) == num_glyphs:
            break

    if len(final_glyphs_list) < num_glyphs:
        print(f"Warning: K-Means resulted in fewer than {num_glyphs} active centroids ({len(final_glyphs_list)}). Filling with random.")
        while len(final_glyphs_list) < num_glyphs:
            final_glyphs_list.append(np.random.rand(block_len).astype(np.float32))

    final_glyphs = np.array(final_glyphs_list) # (num_glyphs, block_len)

    # Now, generate the block_sequence for the original video using the finalized glyphs
    print("Generating block sequence for rendering...")
    final_glyphs_flat = final_glyphs.reshape(num_glyphs, block_len)
    block_sequence_for_rendering = []
    
    current_patch_idx = 0
    for frame_shape in tqdm(frames_shape, desc="Mapping video frames to glyphs"):
        frame_h, frame_w = frame_shape
        num_blocks_h = frame_h // block_h
        num_blocks_w = frame_w // block_w
        num_patches_in_frame = num_blocks_h * num_blocks_w

        frame_patches = all_patches[current_patch_idx : current_patch_idx + num_patches_in_frame]
        best_glyph_indices_frame = _find_best_glyph_for_patches(frame_patches, final_glyphs_flat)
        block_sequence_for_rendering.append(best_glyph_indices_frame)

        current_patch_idx += num_patches_in_frame
    
    # Reshape final glyphs to 3D for consistency (num_glyphs, block_h, block_w)
    final_glyphs_3d = final_glyphs.reshape(num_glyphs, block_h, block_w)

    # Save the trained blocks and metadata
    project_data = {
        'blocks': torch.from_numpy(final_glyphs_3d).float(), # Convert back to torch tensor for consistency with ML path
        'width': width,
        'height': height,
        'block_size': block_size,
        'num_glyphs': num_glyphs,
        'block_sequence': block_sequence_for_rendering, # Add the generated block sequence
        'original_fps': original_fps # Add original FPS
    }
    with open(output_rc_path, 'wb') as f:
        pickle.dump(project_data, f)
    print(f"K-Means project saved to {output_rc_path}")
