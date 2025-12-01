import lzma
import numpy as np
import torch
import struct

class RottenCompressor:
    @staticmethod
    def save_project(file_path, blocks, block_sequence, metadata):
        """
        Saves the project data (blocks, block_sequence, metadata) to a compressed .rc file.

        :param file_path: Path to save the .rc file.
        :param blocks: Blocks data (Tensor/Numpy), expected to be monochromatic [0, 1] range.
        :param block_sequence: List of frame indices.
        :param metadata: Dictionary of project metadata.
        """
        if isinstance(blocks, torch.Tensor):
            blocks = blocks.cpu().numpy()

        # Convert blocks to 1-bit or 8-bit packed binary format
        # Assuming blocks are monochromatic and values are 0 or 1
        # For 1-bit: each block should be (H, W) and values are 0 or 1
        # For 8-bit: each block should be (H, W) and values are 0-255 (after scaling)

        # Determine block dimensions from the first block
        if blocks.ndim == 4: # (num_blocks, 1, H, W)
            _, _, block_h, block_w = blocks.shape
        elif blocks.ndim == 3: # (num_blocks, H, W)
            block_h, block_w = blocks.shape[1:]
        else:
            raise ValueError("Unsupported blocks dimension. Expected 3 (num_blocks, H, W) or 4 (num_blocks, 1, H, W).")


        # For simplicity, let's target 8-bit grayscale for now.
        # If true 1-bit packing is required, it will be more complex.
        # Scale blocks from [0, 1] to [0, 255] and convert to uint8
        blocks_packed = (blocks * 255).astype(np.uint8)

        # Flatten the block_sequence into a tight integer array
        # Determine the smallest uint type for block_sequence
        max_block_idx = blocks.shape[0] - 1 if blocks.shape[0] > 0 else 0
        if max_block_idx < 2**8:
            block_sequence_dtype = np.uint8
        elif max_block_idx < 2**16:
            block_sequence_dtype = np.uint16
        elif max_block_idx < 2**32:
            block_sequence_dtype = np.uint32
        else:
            block_sequence_dtype = np.uint64
        block_sequence_flat = np.array(block_sequence, dtype=block_sequence_dtype)

        # Prepare metadata for serialization (convert torch tensors to numpy/primitive types)
        serializable_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, torch.Tensor):
                serializable_metadata[k] = v.cpu().numpy().tolist() # Convert to list for JSON compatibility
            elif isinstance(v, np.ndarray):
                serializable_metadata[k] = v.tolist()
            else:
                serializable_metadata[k] = v

        # Combine all data into a dictionary
        data_to_compress = {
            "blocks": blocks_packed.tobytes(), # Store as bytes
            "blocks_shape": blocks_packed.shape,
            "block_sequence": block_sequence_flat.tobytes(), # Store as bytes
            "block_sequence_dtype": block_sequence_flat.dtype.str,
            "metadata": serializable_metadata,
        }

        # Serialize using pickle and then compress with LZMA
        # Using struct for metadata like shape and dtype prefix for self-describing format
        # This will need a more robust serialization than simple pickle for long term stability
        # For now, we'll use a simple structure:
        # metadata_len (4 bytes) | metadata_json | blocks_len (4 bytes) | blocks_data | block_sequence_len (4 bytes) | block_sequence_data
        
        # A more robust approach might be to use a self-describing format like MessagePack or a custom header.
        # For this task, let's simplify and serialize the entire dictionary using pickle, then compress.
        # This is not ideal for cross-language compatibility or versioning, but fulfills the request for LZMA compression.
        
        import pickle
        pickled_data = pickle.dumps(data_to_compress)
        compressed_data = lzma.compress(pickled_data)

        with open(file_path, "wb") as f:
            f.write(compressed_data)

    @staticmethod
    def load_project(file_path):
        """
        Loads project data from a compressed .rc file.

        :param file_path: Path to the .rc file.
        :return: (blocks, block_sequence, metadata)
        """
        with open(file_path, "rb") as f:
            compressed_data = f.read()

        decompressed_data = lzma.decompress(compressed_data)
        import pickle
        data_loaded = pickle.loads(decompressed_data)

        blocks_packed_bytes = data_loaded["blocks"]
        blocks_shape = data_loaded["blocks_shape"]
        block_sequence_bytes = data_loaded["block_sequence"]
        block_sequence_dtype_str = data_loaded["block_sequence_dtype"]
        metadata = data_loaded["metadata"]

        # Reconstruct blocks
        blocks_packed = np.frombuffer(blocks_packed_bytes, dtype=np.uint8).reshape(blocks_shape)
        blocks = blocks_packed.astype(np.float32) / 255.0 # Scale back to [0, 1]

        # Reconstruct block_sequence
        block_sequence_dtype = np.dtype(block_sequence_dtype_str)
        block_sequence_flat = np.frombuffer(block_sequence_bytes, dtype=block_sequence_dtype)

        # Convert metadata back to original types if necessary (e.g., list back to numpy array if it was one)
        # This part might need refinement depending on the actual metadata structure.
        # For simplicity, assuming metadata values are directly usable after loading or can be converted to torch tensors as needed.

        # Example: Convert lists back to torch tensors if they represent block_shape or similar configs
        loaded_metadata = {}
        for k, v in metadata.items():
            if k == "block_shape" and isinstance(v, list): # Assuming block_shape might be stored as list
                loaded_metadata[k] = torch.tensor(v, dtype=torch.int32)
            else:
                loaded_metadata[k] = v
        
        # Ensure blocks are returned as torch.Tensor if that's the expected usage later
        blocks_tensor = torch.from_numpy(blocks).float()

        return blocks_tensor, block_sequence_flat.tolist(), loaded_metadata

