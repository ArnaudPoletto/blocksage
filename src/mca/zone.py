import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.utils.block_dictionary import (
    get_block_id_dictionary,
    get_block_color_dictionary,
)
from src.config import (
    CHUNK_XZ_SIZE,
    SECTION_SIZE,
    AIR_NAME,
    CAVE_AIR_NAME,
    VOID_AIR_NAME,
    BLACK_COLOR,
)


class Zone:
    """A zone as a collection of blocks."""

    def __init__(
        self, data: np.ndarray, x_world: int, y_world: int, z_world: int
    ) -> None:
        """
        Initialize a zone.

        Args:
            data (np.ndarray): The array containing the block indices.
            x_world (int): The x coordinate of the zone in the world.
            z_world (int): The z coordinate of the zone in the world.
        """
        self.data = data
        self.x_world = x_world
        self.y_world = y_world
        self.z_world = z_world

    def get_number_of_blocks(self) -> int:
        """
        Returns the number of blocks in the zone.

        Returns:
            int: The number of blocks in the zone.
        """
        return np.prod(self.data.shape)

    def get_world_coordinates(self) -> tuple:
        """
        Returns the world coordinates of the zone.

        Returns:
            tuple: The world coordinates of the zone.
        """
        return self.x_world, self.y_world, self.z_world

    @abstractmethod
    def get_data_for_display(self) -> np.ndarray:
        """
        Returns the data of the zone with the view applied, for display purposes.
        """
        pass

    def get_data(self):
        """
        Returns the data of the zone.
        """
        return self.data

    @staticmethod
    def _get_first_non_air_rgb(
        zone: np.ndarray, block_id_dict: dict, id_color_dict: dict
    ) -> np.ndarray:
        air_block_id = block_id_dict[AIR_NAME]
        cave_air_block_id = block_id_dict[CAVE_AIR_NAME]
        void_air_block_id = block_id_dict[VOID_AIR_NAME]

        # Get the first non-air block for each xz slice
        non_air_mask = (
            (zone != air_block_id)
            & (zone != cave_air_block_id)
            & (zone != void_air_block_id)
            & (zone != np.uint16(-1))
        )
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

        # Add depth to the rgb values
        first_non_air_rgb += (first_non_air_indices[:, :, None] % 2 == 0) * 10
        first_non_air_rgb = np.clip(first_non_air_rgb, 0, 255).astype(np.uint8)

        return first_non_air_rgb

    def display(
        self, block_id_dict: dict = None, block_color_dict: dict = None
    ) -> None:
        """
        Display the region of blocks.

        Args:
            block_id_dict (dict, optional): Dictionary mapping block names to block ids. Defaults to None.
            block_color_dict (dict, optional): Dictionary mapping block names to rgb values. Defaults to None.
        """
        # Get dictionaries if not specified
        if block_id_dict is None:
            block_id_dict = get_block_id_dictionary()
        if block_color_dict is None:
            block_color_dict = get_block_color_dictionary()

        id_color_dict = {
            block_id: block_color_dict[block_name]
            for block_name, block_id in block_id_dict.items()
            if block_name in block_color_dict
        }

        # XZ view
        zone_xz = self.get_data_for_display()
        first_non_air_rgb_xz = Zone._get_first_non_air_rgb(
            zone_xz, block_id_dict, id_color_dict
        )

        # XY view
        zone_xy = self.get_data_for_display().transpose((0, 2, 1))
        first_non_air_rgb_xy = Zone._get_first_non_air_rgb(
            zone_xy, block_id_dict, id_color_dict
        )

        # ZY view
        zone_zy = self.get_data_for_display().transpose((2, 0, 1))
        first_non_air_rgb_zy = Zone._get_first_non_air_rgb(
            zone_zy, block_id_dict, id_color_dict
        )

        # Plot
        gs = GridSpec(1, 3)
        plt.figure(figsize=(15, 15))

        # Add the first XZ view on top and center
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(first_non_air_rgb_xz.transpose((1, 0, 2)), origin="lower")
        ax1.set_title("Map of first non-air blocks (xz)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("z")
        ax1.set_xticks(np.arange(zone_xz.shape[0], step=CHUNK_XZ_SIZE))
        ax1.set_xticklabels(
            np.arange(self.x_world, self.x_world + zone_xz.shape[0], CHUNK_XZ_SIZE),
            rotation=90,
        )
        ax1.set_yticks(np.arange(zone_xz.shape[2], step=CHUNK_XZ_SIZE))
        ax1.set_yticklabels(
            np.arange(self.z_world, self.z_world + zone_xz.shape[2], CHUNK_XZ_SIZE),
            rotation=0,
        )

        # Add the XY view on bottom left
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(first_non_air_rgb_xy.transpose((1, 0, 2)), origin="lower")
        ax2.set_title("Map of first non-air blocks (xy)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_xticks(np.arange(zone_xy.shape[0], step=CHUNK_XZ_SIZE))
        ax2.set_xticklabels(
            np.arange(self.x_world, self.x_world + zone_xy.shape[0], CHUNK_XZ_SIZE),
            rotation=90,
        )
        ax2.set_yticks(np.arange(zone_xy.shape[2], step=SECTION_SIZE))
        ax2.set_yticklabels(
            np.arange(self.y_world, self.y_world + zone_xy.shape[2], SECTION_SIZE),
            rotation=0,
        )

        # Add the YZ view on bottom right
        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(first_non_air_rgb_zy.transpose((1, 0, 2)), origin="lower")
        ax3.set_title("Map of first non-air blocks (zy)")
        ax3.set_xlabel("z")
        ax3.set_ylabel("y")
        ax3.set_xticks(np.arange(zone_zy.shape[0], step=CHUNK_XZ_SIZE))
        ax3.set_xticklabels(
            np.arange(self.z_world, self.z_world + zone_zy.shape[0], CHUNK_XZ_SIZE),
            rotation=90,
        )
        ax3.set_yticks(np.arange(zone_zy.shape[2], step=SECTION_SIZE))
        ax3.set_yticklabels(
            np.arange(self.y_world, self.y_world + zone_zy.shape[2], SECTION_SIZE),
            rotation=0,
        )

        # Adjust layout
        plt.tight_layout()
        plt.show()