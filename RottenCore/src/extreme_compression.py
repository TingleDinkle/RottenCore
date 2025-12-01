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

    @staticmethod
    def decode(table_bytes, stream_bytes, bit_length):
        """
        Decodes a Huffman bitstream back into a list of integers.
        """
        # 1. Reconstruct Canonical Codes from Table
        try:
            table_data = lzma.decompress(table_bytes)
            symbol_lengths = pickle.loads(table_data)
        except Exception as e:
            raise ValueError(f"Failed to decompress/load Huffman table: {e}")

        sorted_symbols = sorted(symbol_lengths.items(), key=lambda x: (x[1], x[0]))
        
        # Map (length, code) -> symbol
        rev_code_map = {}
        current_code = 0
        current_len = 0
        
        for symbol, length in sorted_symbols:
            if length > current_len:
                current_code <<= (length - current_len)
                current_len = length
            
            rev_code_map[(current_len, current_code)] = symbol
            current_code += 1
            
        # 2. Decode Stream
        decoded_symbols = []
        
        # Bit reader state
        curr_code_val = 0
        curr_code_len = 0
        bits_read = 0
        
        # Iterate through bytes
        for byte_val in stream_bytes:
            if bits_read >= bit_length:
                break
                
            # Process 8 bits from MSB to LSB
            for i in range(7, -1, -1):
                if bits_read >= bit_length:
                    break
                
                bit = (byte_val >> i) & 1
                
                curr_code_val = (curr_code_val << 1) | bit
                curr_code_len += 1
                
                if (curr_code_len, curr_code_val) in rev_code_map:
                    decoded_symbols.append(rev_code_map[(curr_code_len, curr_code_val)])
                    curr_code_val = 0
                    curr_code_len = 0
                
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

    @staticmethod
    def load_project(file_path):
        """
        Loads project data from an .rcx file.
        """
        with open(file_path, "rb") as f:
            data = f.read()
            
        offset = 0
        
        # 1. Header (12 bytes)
        magic, width, height, num_frames = struct.unpack('<4sHHI', data[offset:offset+12])
        offset += 12
        
        if magic != ExtremeCompressor.MAGIC:
            raise ValueError(f"Invalid RCX file. Magic: {magic}")
            
        # 2. Blocks (4096 bytes for 256 blocks)
        # 256 blocks * 16 bytes per block = 4096 bytes
        blocks_len = 4096
        blocks_data = data[offset:offset+blocks_len]
        offset += blocks_len
        
        # Unpack blocks
        # Each byte contains 4 pixels (2 bits each)
        # We need to unpack to (256, 8, 8)
        
        # Create a lookup table or just bit shift
        # Vectorized unpacking with numpy
        blocks_uint8 = np.frombuffer(blocks_data, dtype=np.uint8)
        
        # Expand bits
        # We have N bytes. We want 4N pixels.
        # pixel 0: (byte >> 6) & 3
        # pixel 1: (byte >> 4) & 3
        # pixel 2: (byte >> 2) & 3
        # pixel 3: byte & 3
        
        p0 = (blocks_uint8 >> 6) & 3
        p1 = (blocks_uint8 >> 4) & 3
        p2 = (blocks_uint8 >> 2) & 3
        p3 = blocks_uint8 & 3
        
        # Stack and flatten
        # blocks_uint8 shape is (4096,)
        # We want to interleave p0, p1, p2, p3
        # Stack: (4, 4096)
        pixels = np.stack([p0, p1, p2, p3], axis=1).flatten() # (16384,)
        
        # Reshape to (256, 8, 8)
        # Total pixels = 256 * 8 * 8 = 16384. Checks out.
        blocks_raw = pixels.reshape(256, 8, 8)
        
        # Scale to [0.0, 1.0]
        blocks_tensor = torch.from_numpy(blocks_raw.astype(np.float32) / 3.0)
        
        # 3. Huffman Table
        table_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        table_bytes = data[offset:offset+table_len]
        offset += table_len
        
        # 4. Stream
        bit_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        stream_bytes = data[offset:]
        
        # Decode Huffman
        rle_symbols = HuffmanEncoder.decode(table_bytes, stream_bytes, bit_len)
        
        # 5. RLE Expansion
        final_sequence = []
        RLE_MARKER = 256
        
        i = 0
        while i < len(rle_symbols):
            sym = rle_symbols[i]
            
            if sym == RLE_MARKER:
                # This shouldn't happen as first element or consecutive markers 
                # based on encoder logic, but handle gracefully
                pass 
            else:
                final_sequence.append(sym)
                # Check if next is marker
                if i + 1 < len(rle_symbols) and rle_symbols[i+1] == RLE_MARKER:
                    count = rle_symbols[i+2]
                    # Append count-1 copies (since we appended one above)
                    final_sequence.extend([sym] * (count - 1))
                    i += 3 # Skip sym, marker, count
                    continue
            
            i += 1
            
        # Reshape sequence to frames
        # num_blocks_per_frame = (width // 8) * (height // 8)
        # We trust num_frames from header
        
        # The renderer expects a list of arrays (one per frame) or a 2D array
        # Let's make it a list of numpy arrays to match standard loader behavior
        block_h, block_w = 8, 8
        num_blocks_x = width // block_w
        num_blocks_y = height // block_h
        num_blocks_per_frame = num_blocks_x * num_blocks_y
        
        seq_np = np.array(final_sequence, dtype=np.int32)
        
        # Safety truncation/padding
        total_expected = num_frames * num_blocks_per_frame
        if len(seq_np) < total_expected:
             print(f"Warning: rcx sequence too short. Padding with 0. ({len(seq_np)} vs {total_expected})")
             seq_np = np.pad(seq_np, (0, total_expected - len(seq_np)), 'constant')
        elif len(seq_np) > total_expected:
             seq_np = seq_np[:total_expected]
             
        block_sequence = seq_np.reshape((num_frames, num_blocks_per_frame))
        
        metadata = {
            'width': width,
            'height': height,
            'block_size': (8, 8),
            'num_glyphs': 256,
            'original_fps': 30.0 # RCX v1 doesn't store FPS currently? Encoder doesn't write it.
            # Checking encoder... "header = struct.pack('<4sHHI', ...)"
            # Indeed, FPS was left out of RCX header in previous step.
            # We will default to 30.0.
        }
        
        return blocks_tensor, block_sequence, metadata