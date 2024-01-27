import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np
import matplotlib.pyplot as plt

from src.utils.view import view_by_region
from src.utils.block_dictionary import (
    get_block_id_dictionary,
    get_block_color_dictionary,
)

AIR_NAME = "air"


def display_region(
    region: np.ndarray,
    block_id_dict: dict = None,
    block_color_dict: dict = None,
    apply_view: bool = False,
):
    """
    Display a region of blocks.

    Args:
        region (np.ndarray): Region of blocks, either of shape (chunk_x, chunk_z, section, section_y, section_z, section_x) = (32, 32, 24, 16, 16, 16) or (region_x, region_y, region_z) = (512, 384, 512)
        block_id_dict (dict, optional): Dictionary mapping block names to block ids. Defaults to None.
        block_color_dict (dict, optional): Dictionary mapping block names to rgb values. Defaults to None.
        apply_view (bool, optional): Whether to change the view, i.e. from (chunk_x, chunk_z, section, section_y, section_z, section_x) to (region_x, region_y, region_z). Defaults to False.
    """
    # Get dictionaries if not specified
    if block_id_dict is None:
        block_id_dict = get_block_id_dictionary()
    if block_color_dict is None:
        block_color_dict = get_block_color_dictionary()

    # Apply view if specified
    if apply_view:
        region = view_by_region(region)

    # Get dictionaries
    id_color_dict = {
        block_id: block_color_dict[block_name]
        for block_name, block_id in block_id_dict.items()
        if block_name in block_color_dict
    }
    air_block_id = block_id_dict[AIR_NAME]

    # Get the first non-air block for each xz slice
    non_air_mask = region != air_block_id
    first_non_air_indices = np.argmax(non_air_mask[:, ::-1, :], axis=1)
    first_non_air_blocks = region[:, ::-1, :][
        np.arange(region.shape[0])[:, None],
        first_non_air_indices,
        np.arange(region.shape[2])[None, :],
    ]

    # Get the rgb values for the first non-air blocks
    first_non_air_r = np.vectorize(lambda x: id_color_dict.get(x)[0])(
        first_non_air_blocks
    )
    first_non_air_g = np.vectorize(lambda x: id_color_dict.get(x)[1])(
        first_non_air_blocks
    )
    first_non_air_b = np.vectorize(lambda x: id_color_dict.get(x)[2])(
        first_non_air_blocks
    )

    first_non_air_rgb = np.stack(
        [first_non_air_r, first_non_air_g, first_non_air_b], axis=-1
    )

    # Display the rgb values
    plt.figure(figsize=(10, 10))
    plt.imshow(first_non_air_rgb[::-1, :, :], origin="lower")
    plt.title("Map of first non-air blocks in each xz slice")
    plt.tight_layout()
    plt.show()
