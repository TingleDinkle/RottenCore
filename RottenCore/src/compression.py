import lzma
import numpy as np
import torch
import struct

class RottenCompressor:
    MAGIC = b'ROTT'
    VERSION = 1

    @staticmethod
    def save_project(file_path, blocks, block_sequence, metadata):
        """
        Saves the project data to a compressed .rc file using a custom binary format.
        
        Format Structure (LZMA Compressed Blob):
        - Header:
            - Magic (4 bytes): 'ROTT'
            - Version (1 byte): 1
            - Width (2 bytes, uint16)
            - Height (2 bytes, uint16)
            - Block Width (1 byte, uint8)
            - Block Height (1 byte, uint8)
            - Num Glyphs (2 bytes, uint16)
            - Num Frames (4 bytes, uint32)
            - FPS (4 bytes, float32)
        - Data:
            - Blocks: (Num Glyphs * Block Width * Block Height) bytes (uint8, 0-255)
            - Block Sequence: (Num Frames * Blocks Per Frame) entries. 
              Entry size depends on Num Glyphs:
                - < 256: 1 byte (uint8)
                - < 65536: 2 bytes (uint16)
                - >= 65536: 4 bytes (uint32)
        """
        if isinstance(blocks, torch.Tensor):
            blocks = blocks.cpu().numpy()

        width = metadata['width']
        height = metadata['height']
        block_size = metadata['block_size']
        num_glyphs = metadata['num_glyphs']
        original_fps = metadata.get('original_fps', 30.0)

        # Determine block dimensions
        if blocks.ndim == 4:
            _, _, block_h, block_w = blocks.shape
        elif blocks.ndim == 3:
            block_h, block_w = blocks.shape[1:]
        else:
            raise ValueError("Unsupported blocks dimension.")

        # Quantize blocks to 8-bit
        blocks_packed = (blocks * 255).astype(np.uint8)
        blocks_bytes = blocks_packed.tobytes()

        # Process Sequence
        # Flatten sequence
        block_sequence_flat = np.array(block_sequence).flatten()
        num_frames = len(block_sequence) # Assuming block_sequence passed is list of frames
        
        # Determine sequence dtype
        if num_glyphs < 256:
            seq_dtype = np.uint8
            seq_fmt = 'B'
        elif num_glyphs < 65536:
            seq_dtype = np.uint16
            seq_fmt = 'H'
        else:
            seq_dtype = np.uint32
            seq_fmt = 'I'
        
        block_sequence_packed = block_sequence_flat.astype(seq_dtype)
        sequence_bytes = block_sequence_packed.tobytes()

        # Build Header
        # Magic (4), Ver (1), W (2), H (2), BW (1), BH (1), NG (2), NF (4), FPS (4)
        # Total Header Size: 4 + 1 + 2 + 2 + 1 + 1 + 2 + 4 + 4 = 21 bytes
        header = struct.pack(
            '<4sBHHBBHIf',
            RottenCompressor.MAGIC,
            RottenCompressor.VERSION,
            width,
            height,
            block_w,
            block_h,
            num_glyphs,
            num_frames,
            original_fps
        )

        full_binary = header + blocks_bytes + sequence_bytes
        
        # Compress
        compressed_data = lzma.compress(full_binary)

        with open(file_path, "wb") as f:
            f.write(compressed_data)

    @staticmethod
    def load_project(file_path):
        """
        Loads project data from a compressed .rc file.
        """
        with open(file_path, "rb") as f:
            compressed_data = f.read()

        try:
            decompressed_data = lzma.decompress(compressed_data)
        except lzma.LZMAError:
             raise ValueError("Failed to decompress file. It might be corrupted or not an LZMA file.")

        # Parse Header
        header_size = 21
        if len(decompressed_data) < header_size:
             raise ValueError("File too short to contain valid header.")

        magic, version, width, height, block_w, block_h, num_glyphs, num_frames, fps = struct.unpack(
            '<4sBHHBBHIf', decompressed_data[:header_size]
        )

        if magic != RottenCompressor.MAGIC:
            # Fallback to legacy pickle loader if magic doesn't match
            # This ensures backward compatibility if needed, or better, just fail safe.
            # For this migration, let's assume we are fully switching.
            raise ValueError(f"Invalid file format. Magic bytes mismatch: {magic}")
        
        if version != RottenCompressor.VERSION:
             raise ValueError(f"Unsupported file version: {version}")

        offset = header_size

        # Parse Blocks
        # Size = num_glyphs * block_h * block_w bytes
        blocks_size = num_glyphs * block_h * block_w
        if len(decompressed_data) < offset + blocks_size:
             raise ValueError("File truncated in blocks section.")

        blocks_raw = np.frombuffer(decompressed_data, dtype=np.uint8, count=blocks_size, offset=offset)
        blocks = blocks_raw.reshape((num_glyphs, block_h, block_w)).astype(np.float32) / 255.0
        offset += blocks_size

        # Parse Sequence
        num_blocks_per_frame = (width // block_w) * (height // block_h)
        total_sequence_len = num_frames * num_blocks_per_frame
        
        if num_glyphs < 256:
            seq_dtype = np.uint8
        elif num_glyphs < 65536:
            seq_dtype = np.uint16
        else:
            seq_dtype = np.uint32
        
        sequence_bytes_len = total_sequence_len * np.dtype(seq_dtype).itemsize
        
        if len(decompressed_data) < offset + sequence_bytes_len:
             raise ValueError("File truncated in sequence section.")
             
        block_sequence_flat = np.frombuffer(decompressed_data, dtype=seq_dtype, count=total_sequence_len, offset=offset)
        
        # Reshape sequence to list of arrays (frames)
        block_sequence = block_sequence_flat.reshape((num_frames, num_blocks_per_frame))

        blocks_tensor = torch.from_numpy(blocks).float()
        
        metadata = {
            'width': width,
            'height': height,
            'block_size': (block_h, block_w), # Tuple (h, w)
            'num_glyphs': num_glyphs,
            'original_fps': fps
        }

        return blocks_tensor, block_sequence, metadata