import lzma
import numpy as np
import torch
import struct
import heapq
import collections
import pickle

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
        
        # Build Heap: (freq, unique_id, left, right, symbol)
        # unique_id is needed to break ties in heap consistently
        heap = []
        unique_id = 0
        for symbol, freq in freqs.items():
            # Symbol is leaf node
            heapq.heappush(heap, (freq, unique_id, None, None, symbol))
            unique_id += 1
        
        # Build Tree
        while len(heap) > 1:
            f1, uid1, l1, r1, sym1 = heapq.heappop(heap)
            f2, uid2, l2, r2, sym2 = heapq.heappop(heap)
            
            # Create parent
            # Symbol is None for internal nodes
            heapq.heappush(heap, (f1 + f2, unique_id, (f1, uid1, l1, r1, sym1), (f2, uid2, l2, r2, sym2), None))
            unique_id += 1
            
        root = heap[0]
        
        # Generate Code Lengths (Canonical Huffman requirement preparation)
        # We map Symbol -> CodeLength
        symbol_lengths = {}
        
        def traverse(node, depth):
            _, _, left, right, symbol = node
            if symbol is not None:
                symbol_lengths[symbol] = depth
            else:
                traverse(left, depth + 1)
                traverse(right, depth + 1)
        
        traverse(root, 0 if len(freqs) > 1 else 1) # Handle single symbol case
        
        # Generate Canonical Codes
        # Sort by length, then by symbol value
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
            
        # Serialize Table (Mapping of Symbol -> Length is enough to reconstruct Canonical)
        # We compress it for "Extreme" size
        table_data = pickle.dumps(symbol_lengths)
        table_bytes = lzma.compress(table_data)
        
        # Generate Bitstream
        bit_buffer = 0
        bit_count = 0
        byte_array = bytearray()
        
        total_bits = 0
        
        for val in data_list:
            code, length = symbol_codes[val]
            
            # Add to buffer (MSB first packing usually, but here simple LSB shifting for stream)
            # Let's stick to a standard: Stream is read bit by bit.
            # We'll push bits.
            
            # Python handles large ints, but packing manually to bytes is faster in chunks.
            # Optimization: String concatenation of bits is slow.
            # Better: Use bitwise ops.
            
            # Simple implementation:
            for i in range(length - 1, -1, -1):
                bit = (code >> i) & 1
                bit_buffer = (bit_buffer << 1) | bit
                bit_count += 1
                
                if bit_count == 8:
                    byte_array.append(bit_buffer)
                    bit_buffer = 0
                    bit_count = 0
            
            total_bits += length

        # Flush remaining
        if bit_count > 0:
            bit_buffer = (bit_buffer << (8 - bit_count))
            byte_array.append(bit_buffer)
            
        return table_bytes, bytes(byte_array), total_bits


class ExtremeCompressor:
    MAGIC = b'RCX0' # Extreme Mode Magic
    VERSION = 1

    @staticmethod
    def save_project(file_path, blocks, block_sequence, metadata):
        """
        Saves the project in 'Extreme' .rcx format.
        Target: ~60-100KB file size.
        
        Format:
        - Header (12 bytes): MAGIC, WIDTH, HEIGHT, NUM_FRAMES
        - Block Section (4096 bytes): 2-bit quantized blocks (256 * 8 * 2 bytes).
        - Huffman Table Section:
            - Len (4 bytes)
            - LZMA Compressed Pickle of {Symbol: Len}
        - Stream Section:
            - Len (4 bytes) [Bit Length of stream, not byte length!]
            - Raw Huffman Stream Bytes
        """
        
        print("⚠️ Extreme Mode Enabled: Forcing 2-bit quantization and Huffman coding.")
        
        if isinstance(blocks, torch.Tensor):
            blocks = blocks.cpu().numpy() # (256, 8, 8)
            
        if blocks.ndim == 4:
             blocks = blocks.squeeze(1) # Handle (256, 1, 8, 8)

        if blocks.shape[0] != 256 or blocks.shape[1] != 8 or blocks.shape[2] != 8:
            print(f"Warning: Extreme mode optimized for 256 8x8 blocks. Current shape: {blocks.shape}. Encoding might fail or be suboptimal.")
            
        width = metadata['width']
        height = metadata['height']
        num_frames = len(block_sequence) # Assumes block_sequence is list of frames, each frame is list of IDs
        
        # 1. Block Compression (2-bit)
        # Quantize [0, 1] -> {0, 1, 2, 3}
        q_blocks = np.clip(np.round(blocks * 3), 0, 3).astype(np.uint8)
        
        # Pack 4 pixels (1 byte)
        # Row 0: p0, p1, p2, p3 -> Byte 0
        # Row 0: p4, p5, p6, p7 -> Byte 1
        # Total 16 bytes per block.
        
        packed_blocks = bytearray()
        for b_idx in range(blocks.shape[0]):
            for row in range(8):
                r = q_blocks[b_idx, row]
                # Byte 1: cols 0-3
                b1 = (r[0] << 6) | (r[1] << 4) | (r[2] << 2) | r[3]
                # Byte 2: cols 4-7
                b2 = (r[4] << 6) | (r[5] << 4) | (r[6] << 2) | r[7]
                packed_blocks.append(b1)
                packed_blocks.append(b2)
        
        blocks_data = bytes(packed_blocks)
        
        # 2. Sequence Compression (RLE + Huffman)
        # Flatten sequence
        flat_seq = np.array(block_sequence).flatten().tolist()
        
        # Generate RLE Stream
        # Symbols: 0-255 (Literals), 256 (RLE Marker), >256 (Not used, counts follow marker)
        # Actually, we will emit [Literal, Marker, Count] where Count is treated as a symbol too.
        
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
            # Flush
            rle_stream.append(curr)
            if count > 1:
                rle_stream.append(RLE_MARKER)
                rle_stream.append(count)
                
        # Huffman Encode
        table_bytes, stream_bytes, bit_len = HuffmanEncoder.encode(rle_stream)
        
        # 3. Construct File
        # Header: MAGIC (4), WIDTH (2), HEIGHT (2), NUM_FRAMES (4)
        header = struct.pack('<4sHHH', ExtremeCompressor.MAGIC, width, height, num_frames) 
        # Note: Previous struct was I for Frames. H is unsigned short (65535). 
        # Let's use I for frames to be safe (4 bytes).
        header = struct.pack('<4sHHI', ExtremeCompressor.MAGIC, width, height, num_frames)

        with open(file_path, 'wb') as f:
            f.write(header)
            f.write(blocks_data) # Fixed 4096 bytes (if 256 blocks)
            
            # Section 2: Table
            f.write(struct.pack('<I', len(table_bytes)))
            f.write(table_bytes)
            
            # Section 3: Stream
            f.write(struct.pack('<I', bit_len)) # Write BIT length
            f.write(stream_bytes)
            
        print(f"Extreme compression complete. Saved to {file_path}")
