import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np

from src.mca.zone import Zone
from src.mca.section import Section
from src.config import CHUNK_XZ_SIZE, MIN_Y


class Chunk(Zone):
    """A chunk of a region. There are 32x32 chunks in a region."""

    def __init__(self, region, x: int, z: int) -> None:
        """
        Initialize a region.

        Args:
            region (Region): The region the chunk belongs to.
            x (int): The x coordinate of the chunk.
            z (int): The z coordinate of the chunk.
        """
        if x < 0 or x >= region.data.shape[0]:
            raise ValueError(f"❌ x must be in [0, {region.data.shape[0]}), not {x}.")
        if z < 0 or z >= region.data.shape[1]:
            raise ValueError(f"❌ z must be in [0, {region.data.shape[1]}), not {z}.")

        x_world = region.x_world + CHUNK_XZ_SIZE * x
        z_world = region.z_world + CHUNK_XZ_SIZE * z
        super().__init__(region.data[x, z], x_world, MIN_Y, z_world)
        self.region = region
        self.x = x
        self.z = z

    def get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_chunk()

    def get_section(self, y: int) -> Section:
        """
        Returns a section of blocks.

        Args:
            y (int): The y coordinate of the section.

        Returns:
            Section: The section of blocks.
        """
        if y < 0 or y >= self.data.shape[0]:
            raise ValueError(f"❌ y must be in [0, {self.data.shape[0]}), not {y}.")

        return Section(self, y)

    def get_data_by_chunk(self) -> np.ndarray:
        """
        View the blocks by chunk, i.e. as an array of shape (chunk_x, chunk_y, chunk_z) = (section_x, section * section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (chunk_x, chunk_y, chunk_z) = (section_x, section * section_y, section_z).
        """
        section, section_y, section_z, section_x = self.data.shape

        return self.data.reshape((section * section_y, section_z, section_x)).transpose(
            (2, 0, 1)
        )

    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (section, section_x, section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (section, section_x, section_y, section_z).
        """
        return self.data.transpose((0, 3, 1, 2))
