import lzma
import numpy as np
import torch
import struct
import heapq
import collections
import pickle
import os

class HuffmanEncoder:
    @staticmethod
    def encode(data_list):
        """
        Encodes a list of integers using Huffman coding.
        Returns:
            - table_bytes: Serialized symbol->code_len table (compressed).
            - packed_bytes: The Huffman bitstream packed into bytes.
            - bit_length: The actual number of bits in the stream.
        """
        if not data_list:
            return b'', b'', 0

        # Calculate frequencies
        freqs = collections.Counter(data_list)
        
        # Build Heap
        heap = []
        unique_id = 0
        for symbol, freq in freqs.items():
            heapq.heappush(heap, (freq, unique_id, None, None, symbol))
            unique_id += 1
        
        # Build Tree
        while len(heap) > 1:
            f1, uid1, l1, r1, sym1 = heapq.heappop(heap)
            f2, uid2, l2, r2, sym2 = heapq.heappop(heap)
            heapq.heappush(heap, (f1 + f2, unique_id, (f1, uid1, l1, r1, sym1), (f2, uid2, l2, r2, sym2), None))
            unique_id += 1
            
        root = heap[0]
        
        # Generate Code Lengths
        symbol_lengths = {}
        def traverse(node, depth):
            _, _, left, right, symbol = node
            if symbol is not None:
                symbol_lengths[symbol] = depth
            else:
                traverse(left, depth + 1)
                traverse(right, depth + 1)
        
        traverse(root, 0 if len(freqs) > 1 else 1)
        
        # Generate Canonical Codes
        sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: (x[1], x[0]))
        
        symbol_codes = {}
        current_code = 0
        current_len = 0
        
        for symbol, length in sorted_symbols:
            if length > current_len:
                current_code <<= (length - current_len)
                current_len = length
            symbol_codes[symbol] = (current_code, length)
            current_code += 1
            
        # Serialize Table
        table_data = pickle.dumps(symbol_lengths)
        table_bytes = lzma.compress(table_data)
        
        # Generate Bitstream
        bit_buffer = 0
        bit_count = 0
        byte_array = bytearray()
        total_bits = 0
        
        for val in data_list:
            code, length = symbol_codes[val]
            for i in range(length - 1, -1, -1):
                bit = (code >> i) & 1
                bit_buffer = (bit_buffer << 1) | bit
                bit_count += 1
                if bit_count == 8:
                    byte_array.append(bit_buffer)
                    bit_buffer = 0
                    bit_count = 0
            total_bits += length

        if bit_count > 0:
            bit_buffer = (bit_buffer << (8 - bit_count))
            byte_array.append(bit_buffer)
            
        return table_bytes, bytes(byte_array), total_bits

    @staticmethod
    def decode(table_bytes, stream_bytes, bit_length):
        """
        Decodes a Huffman bitstream.
        """
        if bit_length == 0:
            return []

        # 1. Reconstruct Canonical Huffman Tree
        symbol_lengths = pickle.loads(lzma.decompress(table_bytes))
        sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: (x[1], x[0]))
        
        # Map (length, code) -> symbol
        code_map = {}
        current_code = 0
        current_len = 0
        
        for symbol, length in sorted_symbols:
            if length > current_len:
                current_code <<= (length - current_len)
                current_len = length
            code_map[(length, current_code)] = symbol
            current_code += 1

        # 2. Parse Bitstream
        decoded_symbols = []
        curr_val = 0
        curr_len = 0
        bits_read = 0
        
        for byte in stream_bytes:
            for i in range(7, -1, -1):
                if bits_read >= bit_length:
                    break
                
                bit = (byte >> i) & 1
                curr_val = (curr_val << 1) | bit
                curr_len += 1
                
                # Check for match
                if (curr_len, curr_val) in code_map:
                    decoded_symbols.append(code_map[(curr_len, curr_val)])
                    curr_val = 0
                    curr_len = 0
                
                bits_read += 1
                
        return decoded_symbols


class ExtremeCompressor:
    MAGIC = b'RCX0' # Extreme Mode Magic
    VERSION = 1

    @staticmethod
    def save_project(file_path, blocks, block_sequence, metadata):
        """
        Saves the project in 'Extreme' .rcx format.
        Target: ~60-100KB file size.
        """
        print("⚠️ Extreme Mode Enabled: Forcing 2-bit quantization and Huffman coding.")
        
        if isinstance(blocks, torch.Tensor):
            blocks = blocks.cpu().numpy()
            
        if blocks.ndim == 4:
             blocks = blocks.squeeze(1)

        if blocks.shape[0] != 256 or blocks.shape[1] != 8 or blocks.shape[2] != 8:
            print(f"Warning: Extreme mode optimized for 256 8x8 blocks. Current shape: {blocks.shape}.")
            
        width = metadata['width']
        height = metadata['height']
        num_frames = len(block_sequence)
        
        # 1. Block Compression (2-bit)
        q_blocks = np.clip(np.round(blocks * 3), 0, 3).astype(np.uint8)
        
        packed_blocks = bytearray()
        for b_idx in range(blocks.shape[0]):
            for row in range(8):
                r = q_blocks[b_idx, row]
                # Pack 8 pixels into 2 bytes (4 pixels per byte)
                b1 = (r[0] << 6) | (r[1] << 4) | (r[2] << 2) | r[3]
                b2 = (r[4] << 6) | (r[5] << 4) | (r[6] << 2) | r[7]
                packed_blocks.append(b1)
                packed_blocks.append(b2)
        
        blocks_data = bytes(packed_blocks)
        
        # 2. Sequence Compression (RLE + Huffman)
        flat_seq = np.array(block_sequence).flatten().tolist()
        
        rle_stream = []
        if flat_seq:
            curr = flat_seq[0]
            count = 1
            RLE_MARKER = 256
            
            for val in flat_seq[1:]:
                if val == curr:
                    count += 1
                else:
                    rle_stream.append(curr)
                    if count > 1:
                        rle_stream.append(RLE_MARKER)
                        rle_stream.append(count)
                    curr = val
                    count = 1
            rle_stream.append(curr)
            if count > 1:
                rle_stream.append(RLE_MARKER)
                rle_stream.append(count)
                
        table_bytes, stream_bytes, bit_len = HuffmanEncoder.encode(rle_stream)
        
        # 3. Construct File
        # Header: MAGIC (4), WIDTH (2), HEIGHT (2), NUM_FRAMES (4)
        header = struct.pack('<4sHHI', ExtremeCompressor.MAGIC, width, height, num_frames)

        with open(file_path, 'wb') as f:
            f.write(header)
            f.write(blocks_data) # Fixed 4096 bytes
            f.write(struct.pack('<I', len(table_bytes)))
            f.write(table_bytes)
            f.write(struct.pack('<I', bit_len))
            f.write(stream_bytes)
            
        print(f"Extreme compression complete. Saved to {file_path}")

    @staticmethod
    def load_project(file_path):
        """
        Loads an 'Extreme' .rcx project file.
        """
        with open(file_path, 'rb') as f:
            # 1. Header
            header_bytes = f.read(12)
            magic, width, height, num_frames = struct.unpack('<4sHHI', header_bytes)
            
            if magic != ExtremeCompressor.MAGIC:
                raise ValueError(f"Invalid RCX file magic: {magic}")
                
            # 2. Blocks
            blocks_data = f.read(4096)
            blocks_np = np.zeros((256, 8, 8), dtype=np.float32)
            
            idx = 0
            for b_idx in range(256):
                for row in range(8):
                    # Read 2 bytes per row
                    b1 = blocks_data[idx]
                    b2 = blocks_data[idx+1]
                    idx += 2
                    
                    # Unpack
                    row_vals = [
                        (b1 >> 6) & 3, (b1 >> 4) & 3, (b1 >> 2) & 3, b1 & 3,
                        (b2 >> 6) & 3, (b2 >> 4) & 3, (b2 >> 2) & 3, b2 & 3
                    ]
                    # Scale 0-3 -> 0.0-1.0
                    for col, val in enumerate(row_vals):
                        blocks_np[b_idx, row, col] = val / 3.0
                        
            # 3. Huffman Table
            table_len_bytes = f.read(4)
            table_len = struct.unpack('<I', table_len_bytes)[0]
            table_bytes = f.read(table_len)
            
            # 4. Stream
            bit_len_bytes = f.read(4)
            bit_len = struct.unpack('<I', bit_len_bytes)[0]
            stream_bytes = f.read() # Read rest
            
            # 5. Decode & Expand
            rle_symbols = HuffmanEncoder.decode(table_bytes, stream_bytes, bit_len)
            
            flat_sequence = []
            i = 0
            RLE_MARKER = 256
            
            while i < len(rle_symbols):
                sym = rle_symbols[i]
                if sym == RLE_MARKER:
                    # Should not happen as first element, but safety check
                    pass 
                else:
                    flat_sequence.append(sym)
                    # Check next for marker
                    if i + 1 < len(rle_symbols) and rle_symbols[i+1] == RLE_MARKER:
                        count = rle_symbols[i+2]
                        # Append sym (count-1) more times (since we appended once already)
                        flat_sequence.extend([sym] * (count - 1))
                        i += 3 # Skip sym, marker, count
                        continue
                i += 1
                
            # Reshape Sequence
            # Blocks per frame
            blocks_per_frame = (width // 8) * (height // 8)
            
            # Slice into frames
            block_sequence = []
            for f_idx in range(num_frames):
                start = f_idx * blocks_per_frame
                end = start + blocks_per_frame
                if end > len(flat_sequence):
                    print("Warning: Stream ended prematurely.")
                    break
                frame_seq = np.array(flat_sequence[start:end], dtype=np.int32)
                block_sequence.append(frame_seq)
                
            metadata = {
                'width': width,
                'height': height,
                'num_glyphs': 256,
                'block_size': (8, 8),
                'original_fps': 30.0 # Header doesn't store FPS, assume default
            }
            
            return torch.from_numpy(blocks_np), block_sequence, metadata