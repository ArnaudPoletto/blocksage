import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np

from src.mca.zone import Zone
from src.mca.chunk import Chunk

class Region(Zone):
    """A region as a collection of blocks of shape (region_x, region_z, section, section_y, section_z, section_x)."""

    SHAPE_SIZE = 6

    def __init__(self, data: np.ndarray, x_world: int, z_world: int) -> None:
        """
        Initialize a region.

        Args:
            data (np.ndarray): The array containing the block indices.
            x_world (int): The x coordinate of the region in the world.
            z_world (int): The z coordinate of the region in the world.
        """
        if len(data.shape) != self.SHAPE_SIZE:
            raise ValueError(
                f"❌ region_blocks must be of shape (region_x, region_z, section, section_y, section_z, section_x), not {data.shape}."
            )

        super().__init__(data, x_world, z_world)

    def get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_region()

    def get_chunk(self, x: int, z: int) -> Chunk:
        """
        Returns a chunk of blocks.

        Args:
            x (int): The x coordinate of the chunk.
            z (int): The z coordinate of the chunk.

        Returns:
            np.ndarray: The chunk of blocks.
        """
        if x < 0 or x >= self.data.shape[0]:
            raise ValueError(f"❌ x must be in [0, {self.data.shape[0]}), not {x}.")
        if z < 0 or z >= self.data.shape[1]:
            raise ValueError(f"❌ z must be in [0, {self.data.shape[1]}), not {z}.")
        
        return Chunk(self, x, z)
    
    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (region_x, region_z, section, section_x, section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (region_x, region_z, section, section_x, section_y, section_z).
        """
        return self.data.transpose((0, 1, 2, 5, 3, 4))


    def get_data_by_chunk(self) -> np.ndarray:
        """
        View the blocks by chunk, i.e. as an array of shape (region_x, region_z, chunk_x, chunk_y, chunk_z) = (region_x, region_z, section_x, section * section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (region_x, region_z, chunk_x, chunk_y, chunk_z) = (region_x, region_z, section_x, section * section_y, section_z).
        """
        region_x, region_z, section, section_y, section_z, section_x = self.data.shape

        return self.data \
            .reshape((region_x, region_z, section * section_y, section_z, section_x)) \
            .transpose((0, 1, 4, 2, 3))


    def get_data_by_region(self) -> np.ndarray:
        """
        View the blocks by region, i.e. as an array of shape (region_x, region_y, region_z) = (region_x * section_x, section * section_y, region_z * section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (region_x, region_y, region_z) = (region_x * section_x, section * section_y, region_z * section_z).
        """
        region_x, region_z, section, section_y, section_z, section_x = self.data.shape

        return (
            self.data.transpose(0, 5, 1, 4, 2, 3)
            .reshape((region_x * section_x, region_z * section_z, section * section_y))
            .transpose(0, 2, 1)
        )