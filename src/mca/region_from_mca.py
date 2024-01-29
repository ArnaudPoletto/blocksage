import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import zlib
import struct
import numpy as np
from tqdm import tqdm
from io import BytesIO
from nbtlib import File
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.log import log
from src.utils.block_dictionary import get_block_id_dictionary
from src.mca.region import Region
from src.config import N_CHUNKS_PER_REGION_PER_DIM, CHUNK_Y_SIZE, MIN_Y, SECTION_SIZE

OFFSET_SHIFT = 8
BITS_PER_LONG = 64
SECTOR_BYTES = 4096
CHUNK_HEADER_SIZE = 5
ZLIB_COMPRESSION_TYPE = 2
SECTOR_COUNT_MASK = 0xFF


TOTAL_SECTION_BLOCKS = SECTION_SIZE * SECTION_SIZE * SECTION_SIZE


def process_section(data: np.ndarray, bit_length: int) -> np.ndarray:
    """
    Process a section of blocks for any bit length, using vectorized operations.

    Args:
        data (np.ndarray): LongArray containing the block indices.
        bit_length (int): Number of bits per block index.

    Returns:
        np.ndarray: Array of block indices.
    """
    if bit_length <= 0 or bit_length > 64:
        raise ValueError(f"Invalid bit length: {bit_length}.")

    indices_per_long = 64 // bit_length

    # Mask and shift the data
    shifts = np.arange(indices_per_long, dtype=np.uint64) * bit_length
    data_reshaped = data.repeat(indices_per_long).reshape(data.shape[0], -1)
    mask = (1 << bit_length) - 1
    masked_data = (data_reshaped >> shifts) & mask

    return masked_data.flatten()[
        :TOTAL_SECTION_BLOCKS
    ]  # Flatten and trim to the correct size (last long may be incomplete)


def _process_chunk(nbt_data: File, block_dict: dict) -> np.ndarray:
    """
    Process a chunk of blocks, i.e. a 24x16x16x16 part of the world.

    Args:
        nbt_data (File): NBT data of the chunk.
        block_dict (dict): Dictionary of block states and their corresponding index.

    Returns:
        int: X coordinate of the chunk in the region.
        int: Z coordinate of the chunk in the region.
        int: X coordinate of the chunk in the world.
        int: Z coordinate of the chunk in the world.
        np.ndarray: Array of block IDs of shape (24, 16, 16, 16).
    """
    chunk_x_in_region = nbt_data["xPos"] % N_CHUNKS_PER_REGION_PER_DIM
    chunk_z_in_region = nbt_data["zPos"] % N_CHUNKS_PER_REGION_PER_DIM
    chunk_x_in_world = nbt_data["xPos"] * SECTION_SIZE
    chunk_z_in_world = nbt_data["zPos"] * SECTION_SIZE
    chunk_blocks = np.zeros(
        (CHUNK_Y_SIZE // SECTION_SIZE, SECTION_SIZE, SECTION_SIZE, SECTION_SIZE),
        dtype=np.uint16,
    )

    for section in nbt_data["sections"]:
        section_palette = np.asarray(
            [
                block["Name"].replace("minecraft:", "")
                for block in section["block_states"]["palette"]
            ]
        )
        section_data = section["block_states"].get("data")
        y = int(section["Y"]) - MIN_Y // SECTION_SIZE

        # If there is no data array, all blocks are the same: the first block in the palette
        if section_data is None:
            section_block_indices = np.zeros(
                (SECTION_SIZE, SECTION_SIZE, SECTION_SIZE), dtype=np.uint16
            )
        else:
            bit_length = max(
                4, int(np.ceil(np.log2(len(section_palette))))
            )  # At least 4 bits, or log2 of palette size
            section_data = np.asarray(section_data, dtype=np.uint64)
            section_block_indices = process_section(section_data, bit_length)

        # Convert block indices to block IDs
        section_blocks = np.vectorize(block_dict.get)(
            section_palette[section_block_indices]
        )

        # Add the section to the chunk
        chunk_blocks[y] = section_blocks.reshape(
            (SECTION_SIZE, SECTION_SIZE, SECTION_SIZE)
        )

    return (
        chunk_x_in_region,
        chunk_z_in_region,
        chunk_x_in_world,
        chunk_z_in_world,
        chunk_blocks,
    )


def _read_and_process_chunk(chunk_data_stream: BytesIO, block_dict: dict) -> np.ndarray:
    """
    Read and process a chunk of blocks, i.e. a 24x16x16x16 part of the world.

    Args:
        chunk_data_stream (BytesIO): Stream of the chunk data.
        block_dict (dict): Dictionary of block states and their corresponding index.

    Returns:
        int: X coordinate of the chunk in the region.
        int: Z coordinate of the chunk in the region.
        int: X coordinate of the chunk in the world.
        int: Z coordinate of the chunk in the world.
        np.ndarray: Array of block IDs of shape (24, 16, 16, 16).
    """

    # Seek the chunk from the offset and sector count
    chunk_data_stream.seek(0)

    # Read the chunk header
    chunk_data_length, compression_type = struct.unpack(
        ">iB", chunk_data_stream.read(CHUNK_HEADER_SIZE)
    )
    # Invalid chunk or unsupported compression type
    if chunk_data_length <= 0 or compression_type != ZLIB_COMPRESSION_TYPE:
        return

    # Read, decompress and convert the chunk data
    chunk_data = chunk_data_stream.read(
        chunk_data_length - 1
    )  # -1 to exclude the compression type byte
    decompressed_data_stream = BytesIO(zlib.decompress(chunk_data))
    with decompressed_data_stream:
        nbt_data = File.parse(decompressed_data_stream)

    return _process_chunk(nbt_data, block_dict)


def get_region(
    file_path: str,
    block_id_dict: dict = None,
    parallelize_chunks: bool = True,
    show_bar: bool = True,
) -> Region:
    """
    Get the block data of a region file.

    Args:
        file_path (str): Path to the region file.
        block_dict (dict): Dictionary of block states and their corresponding index.
        parallelize_chunks (bool): Whether to parallelize chunk processing. Defaults to True.
        show_bar (bool): Whether to show a progress bar. Defaults to True.

    Returns:
        Region: A region of blocks.
    """
    if block_id_dict is None:
        block_id_dict = get_block_id_dictionary()

    # Read the region file
    with open(file_path, "rb") as region_file:
        region_data = region_file.read()
    locations = struct.unpack(f">{SECTOR_BYTES // 4}I", region_data[:SECTOR_BYTES])

    # Process chunks
    data = (
        np.zeros(
            (
                N_CHUNKS_PER_REGION_PER_DIM,
                N_CHUNKS_PER_REGION_PER_DIM,
                CHUNK_Y_SIZE // SECTION_SIZE,
                SECTION_SIZE,
                SECTION_SIZE,
                SECTION_SIZE,
            ),
            dtype=np.uint16,
        )
        - 1
    )  # -1 for missing sections

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        region_x_world = None
        region_z_world = None
        if show_bar:
            bar = tqdm(
                total=N_CHUNKS_PER_REGION_PER_DIM * N_CHUNKS_PER_REGION_PER_DIM,
                desc="🔄 Processing chunk",
            )
        for chunk_idx in range(
            N_CHUNKS_PER_REGION_PER_DIM * N_CHUNKS_PER_REGION_PER_DIM
        ):
            if locations[chunk_idx] == 0:
                if show_bar:
                    bar.update(1)
                continue

            # Get the chunk data as a stream
            offset = (locations[chunk_idx] >> OFFSET_SHIFT) * SECTOR_BYTES
            sector_count = locations[chunk_idx] & SECTOR_COUNT_MASK
            chunk_data_stream = BytesIO(
                region_data[offset : offset + sector_count * SECTOR_BYTES]
            )

            # Process the chunk, either in parallel or sequentially
            if parallelize_chunks:
                futures.append(
                    executor.submit(
                        _read_and_process_chunk, chunk_data_stream, block_id_dict
                    )
                )
            else:
                (
                    chunk_x_region,
                    chunk_z_region,
                    chunk_x_world,
                    chunk_z_world,
                    chunk_blocks,
                ) = _read_and_process_chunk(chunk_data_stream, block_id_dict)
                data[chunk_x_region, chunk_z_region] = chunk_blocks
                if show_bar:
                    bar.update(1)

                if chunk_x_region == 0 and chunk_z_region == 0:
                    region_x_world = chunk_x_world
                    region_z_world = chunk_z_world

        # Wait for all chunks to finish processing if parallelized
        for future in as_completed(futures):
            try:
                (
                    chunk_x_region,
                    chunk_z_region,
                    chunk_x_world,
                    chunk_z_world,
                    chunk_blocks,
                ) = future.result()
                data[chunk_x_region, chunk_z_region] = chunk_blocks
                if show_bar:
                    bar.update(1)

                if chunk_x_region == 0 and chunk_z_region == 0:
                    region_x_world = chunk_x_world
                    region_z_world = chunk_z_world
            except Exception as e:
                log(f"❌ Error while processing chunk: {e}")

        if show_bar:
            bar.close()

    return Region(data, region_x_world, region_z_world)
