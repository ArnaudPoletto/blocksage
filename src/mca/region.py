import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np

from src.mca.zone import Zone

class Region(Zone):
    """A region as a collection of blocks of shape (region_x, region_z, section, section_y, section_z, section_x)."""

    SHAPE_SIZE = 6

    def __init__(self, data: np.ndarray) -> None:
        """
        Initialize a region.

        Args:
            data (np.ndarray): The array containing the block indices.
        """
        if len(data.shape) != self.SHAPE_SIZE:
            raise ValueError(
                f"❌ region_blocks must be of shape (region_x, region_z, section, section_y, section_z, section_x), not {data.shape}."
            )

        super().__init__(data)

    def _get_data_for_display(self):
        return self.view_by_region()
    
    def view_by_section(self):
        """
        View the blocks by section, i.e. as an array of shape (region_x, region_z, section, section_x, section_y, section_z).

        Args:
            region_blocks (np.ndarray): Array of block IDs of shape (region_x, region_z, section, section_y, section_z, section_x).

        Returns:
            np.ndarray: Array of block IDs of shape (region_x, region_z, section, section_x, section_y, section_z).
        """
        return self.data.transpose((0, 1, 2, 5, 3, 4))


    def view_by_chunk(self):
        """
        View the blocks by chunk, i.e. as an array of shape (region_x, region_z, chunk_x, chunk_y, chunk_z) = (region_x, region_z, section_x, section * section_y, section_z).

        Args:
            region_blocks (np.ndarray): Array of block IDs of shape (region_x, region_z, section, section_y, section_z, section_x).

        Returns:
            np.ndarray: Array of block IDs of shape (region_x, region_z, chunk_x, chunk_y, chunk_z) = (region_x, region_z, section_x, section * section_y, section_z).
        """
        region_x, region_z, section, section_y, section_z, section_x = self.data.shape

        return self.data \
            .reshape((region_x, region_z, section * section_y, section_z, section_x)) \
            .transpose((0, 1, 4, 2, 3))


    def view_by_region(self):
        """
        View the blocks by region, i.e. as an array of shape (region_x, region_y, region_z) = (region_x * section_x, section * section_y, region_z * section_z).

        Args:
            region_blocks (np.ndarray): Array of block IDs of shape (region_x, region_z, section, section_y, section_z, section_x).

        Returns:
            np.ndarray: Array of block IDs of shape (region_x, region_y, region_z) = (region_x * section_x, section * section_y, region_z * section_z).
        """
        region_x, region_z, section, section_y, section_z, section_x = self.data.shape

        return (
            self.data.transpose(0, 5, 1, 4, 2, 3)
            .reshape((region_x * section_x, region_z * section_z, section * section_y))
            .transpose(0, 2, 1)
        )