import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import zlib
import struct
import numpy as np
from tqdm import tqdm
from io import BytesIO
from nbtlib import File
from nbtlib.tag import LongArray
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.logs import log
from src.utils.block_dictionary import get_block_id_dictionary

N_REGIONS = 32
OFFSET_SHIFT = 8
SECTION_SIZE = 16
BITS_PER_LONG = 64
SECTOR_BYTES = 4096
CHUNK_HEADER_SIZE = 5
ZLIB_COMPRESSION_TYPE = 2
SECTOR_COUNT_MASK = 0xFF

MIN_WORLD_HEIGHT = -64
MAX_WORLD_HEIGHT = 320
WORLD_HEIGHT = MAX_WORLD_HEIGHT - MIN_WORLD_HEIGHT


def _process_section(data: LongArray, bit_length: int) -> np.ndarray:
    """
    Process a section of blocks, i.e. a 16x16x16 cube of blocks.

    Args:
        data (LongArray): LongArray containing the block indices.
        bit_length (int): Number of bits per block index.

    Returns:
        np.ndarray: Array of block indices of shape (16, 16, 16).
    """

    total_blocks = SECTION_SIZE * SECTION_SIZE * SECTION_SIZE
    block_indices = np.zeros(total_blocks, dtype=np.uint16) # 16-bit unsigned integers are enough for block indices

    bit_mask = (1 << bit_length) - 1
    bit_offset = 0
    long_index = 0
    start_offset = 0
    for block_idx in range(total_blocks):
        block_indices[block_idx] = (data[long_index] >> start_offset) & bit_mask

        # Since 1.16, block indices do not span multiple longs
        if start_offset + bit_length + bit_length > BITS_PER_LONG:
            remaining_bits = (start_offset + bit_length) - BITS_PER_LONG
            bit_offset += remaining_bits
            long_index += 1
            start_offset = 0
        else:
            bit_offset += bit_length
            start_offset += bit_length

    return block_indices


def _process_chunk(nbt_data: File, block_dict: dict) -> np.ndarray:
    """
    Process a chunk of blocks, i.e. a 24x16x16x16 part of the world.

    Args:
        nbt_data (File): NBT data of the chunk.
        block_dict (dict): Dictionary of block states and their corresponding index.

    Returns:
        np.ndarray: Array of block IDs of shape (24, 16, 16, 16).
    """
    chunk_blocks = np.zeros((WORLD_HEIGHT // SECTION_SIZE, SECTION_SIZE, SECTION_SIZE, SECTION_SIZE), dtype=np.uint16)

    for section in nbt_data['sections']:
        section_palette = np.asarray([block['Name'].replace('minecraft:', '') for block in section['block_states']['palette']])
        section_data = section['block_states'].get('data')
        y = int(section['Y']) - MIN_WORLD_HEIGHT // SECTION_SIZE

        # If there is no data array, all blocks are the same: the first block in the palette
        if section_data is None:
            section_block_indices = np.zeros((SECTION_SIZE, SECTION_SIZE, SECTION_SIZE), dtype=np.uint16)
        else:
            section_data = LongArray(section_data)
            bit_length = max(4, int(np.ceil(np.log2(len(section_palette)))))  # At least 4 bits, or log2 of palette size
            section_block_indices = _process_section(section_data, bit_length)

        # Convert block indices to block IDs
        section_blocks = np.vectorize(block_dict.get)(section_palette[section_block_indices])

        # Add the section to the chunk
        chunk_blocks[y] = section_blocks.reshape((SECTION_SIZE, SECTION_SIZE, SECTION_SIZE))

    return chunk_blocks


def _read_and_process_chunk(chunk_data_stream: BytesIO, block_dict: dict, x: int = None, y: int = None) -> np.ndarray:
    """
    Read and process a chunk of blocks, i.e. a 24x16x16x16 part of the world.

    Args:
        chunk_data_stream (BytesIO): Stream of the chunk data.
        block_dict (dict): Dictionary of block states and their corresponding index.
        x (int): X coordinate of the chunk. Defaults to None.
        y (int): Y coordinate of the chunk. Defaults to None.

    Returns:
        np.ndarray: Array of block IDs of shape (24, 16, 16, 16).
        x (int): X coordinate of the chunk. Not returned if x or y is None.
        y (int): Y coordinate of the chunk. Not returned if x or y is None.
    """

    # Seek the chunk from the offset and sector count
    chunk_data_stream.seek(0)

    # Read the chunk header
    chunk_data_length, compression_type = struct.unpack('>iB', chunk_data_stream.read(CHUNK_HEADER_SIZE))
    # Invalid chunk or unsupported compression type
    if chunk_data_length <= 0 or compression_type != ZLIB_COMPRESSION_TYPE:
        return

    # Read, decompress and convert the chunk data
    chunk_data = chunk_data_stream.read(chunk_data_length - 1)  # -1 to exclude the compression type byte
    decompressed_data_stream = BytesIO(zlib.decompress(chunk_data))
    with decompressed_data_stream:
        nbt_data = File.parse(decompressed_data_stream)

    chunk_blocks = _process_chunk(nbt_data, block_dict)
    if x is not None and y is not None:
        return chunk_blocks, x, y
    
    return _process_chunk(nbt_data, block_dict)


def get_region_data(file_path: str, block_id_dict: dict = None, parallelize_chunks: bool = True):
    """
    Get the block data of a region file.

    Args:
        file_path (str): Path to the region file.
        block_dict (dict): Dictionary of block states and their corresponding index.
        parallelize_chunks (bool): Whether to parallelize chunk processing. Defaults to True.

    Returns:
        np.ndarray: Array of block IDs of shape (32, 32, 24, 16, 16, 16).
    """
    if block_id_dict is None:
        block_id_dict = get_block_id_dictionary()
    
    # Read the region file
    with open(file_path, 'rb') as region_file:
        region_data = region_file.read()
    locations = struct.unpack(f'>{SECTOR_BYTES // 4}I', region_data[:SECTOR_BYTES])

    # Process chunks
    region_blocks = np.zeros((N_REGIONS, N_REGIONS, WORLD_HEIGHT // SECTION_SIZE, SECTION_SIZE, SECTION_SIZE, SECTION_SIZE), dtype=np.uint16)
    futures = []
    bar = tqdm(total=N_REGIONS * N_REGIONS, desc='🔄 Processing chunks')
    if parallelize_chunks:
        executor = ProcessPoolExecutor()

    for chunk_idx in range(N_REGIONS * N_REGIONS):
        x = chunk_idx % N_REGIONS
        y = chunk_idx // N_REGIONS

        if locations[chunk_idx] == 0:
            bar.update(1)
            continue

        # Get the chunk data as a stream
        offset = (locations[chunk_idx] >> OFFSET_SHIFT) * SECTOR_BYTES
        sector_count = locations[chunk_idx] & SECTOR_COUNT_MASK
        chunk_data_stream = BytesIO(region_data[offset:offset + sector_count * SECTOR_BYTES])

        # Process the chunk, either in parallel or sequentially
        if parallelize_chunks:
            futures.append((executor.submit(_read_and_process_chunk, chunk_data_stream, block_id_dict, x, y)))
        else:
            chunk_blocks = _read_and_process_chunk(chunk_data_stream, block_id_dict)
            region_blocks[x, y] = chunk_blocks
            bar.update(1)

    # Wait for all chunks to finish processing if parallelized
    for future in as_completed(futures):
        try:
            chunk_blocks, x, y = future.result()
            region_blocks[x, y] = chunk_blocks
            bar.update(1)
        except Exception as e:
            log(f"❌ Error while processing chunk: {e}")

    bar.close()

    return region_blocks