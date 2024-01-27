import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt

from src.utils.block_dictionary import (
    get_block_id_dictionary,
    get_block_color_dictionary,
)

AIR_NAME = "air"
BLACK_COLOR = [0, 0, 0]


class Zone:
    """A zone as a collection of blocks."""

    def __init__(self, data: np.ndarray) -> None:
        """
        Initialize a zone.

        Args:
            data (np.ndarray): The array containing the block indices.
        """
        self.data = data

    def get_number_of_blocks(self) -> int:
        """
        Returns the number of blocks in the zone.

        Returns:
            int: The number of blocks in the zone.
        """
        return np.prod(self.data.shape)

    @abstractmethod
    def get_data_for_display(self) -> np.ndarray:
        """
        Returns the data of the zone with the view applied, for display purposes.
        """
        pass

    def display(
        self, block_id_dict: dict = None, block_color_dict: dict = None
    ) -> None:
        """
        Display a region of blocks.

        Args:
            region (np.ndarray): Region of blocks, either of shape (chunk_x, chunk_z, section, section_x, section * section_y, section_z)
            block_id_dict (dict, optional): Dictionary mapping block names to block ids. Defaults to None.
            block_color_dict (dict, optional): Dictionary mapping block names to rgb values. Defaults to None.
        """
        # Get dictionaries if not specified
        if block_id_dict is None:
            block_id_dict = get_block_id_dictionary()
        if block_color_dict is None:
            block_color_dict = get_block_color_dictionary()

        # Apply view if specified
        zone = self.get_data_for_display()

        # Get dictionaries
        id_color_dict = {
            block_id: block_color_dict[block_name]
            for block_name, block_id in block_id_dict.items()
            if block_name in block_color_dict
        }
        air_block_id = block_id_dict[AIR_NAME]

        # Get the first non-air block for each xz slice
        non_air_mask = zone != air_block_id
        first_non_air_indices = np.argmax(non_air_mask[:, ::-1, :], axis=1)
        first_non_air_blocks = zone[:, ::-1, :][
            np.arange(zone.shape[0])[:, None],
            first_non_air_indices,
            np.arange(zone.shape[2])[None, :],
        ]

        # Get the rgb values for the first non-air blocks
        first_non_air_r = np.vectorize(lambda x: id_color_dict.get(x, BLACK_COLOR)[0])(
            first_non_air_blocks
        )
        first_non_air_g = np.vectorize(lambda x: id_color_dict.get(x, BLACK_COLOR)[1])(
            first_non_air_blocks
        )
        first_non_air_b = np.vectorize(lambda x: id_color_dict.get(x, BLACK_COLOR)[2])(
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
