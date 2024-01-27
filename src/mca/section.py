import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

from src.mca.zone import Zone
import numpy as np


class Section(Zone):
    """A section of a chunk. There are 24 chunks in a region."""

    SECTION_SIZE = 16

    def __init__(self, chunk, y: int) -> None:
        """
        Initialize a region.

        Args:
            chunk (Chunl): The chunk the section belongs to.
            y (int): The y coordinate of the section.
        """
        if y < 0 or y >= chunk.data.shape[1]:
            raise ValueError(f"❌ y must be in [0, {chunk.data.shape[1]}), not {y}.")

        super().__init__(chunk.data[y])
        self.chunk = chunk
        self.y = y

    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (section_x, section_y, section_z).

        Args:
            region_blocks (np.ndarray): Array of block IDs of shape (section_y, section_z, section_x).

        Returns:
            np.ndarray: Array of block IDs of shape (section_x, section_y, section_z).
        """
        return self.data.transpose((3, 1, 2))
    
    def get_data_for_display(self) -> np.ndarray:
        region_y = self.SECTION_SIZE * self.y
        return self.chunk.get_data_for_display()[:, region_y:region_y + self.SECTION_SIZE:, ]
