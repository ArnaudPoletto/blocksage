import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np

from src.mca.zone import Zone
from src.config import SECTION_SIZE


class Section(Zone):
    """A section of a chunk. There are 24 sections in a chunk."""

    def __init__(self, chunk, y: int) -> None:
        """
        Initialize a section.

        Args:
            chunk (Chunl): The chunk the section belongs to.
            y (int): The y coordinate of the section.
        """
        if y < 0 or y >= chunk.data.shape[0]:
            raise ValueError(f"❌ y must be in [0, {chunk.data.shape[0]}), not {y}.")

        super().__init__(chunk.data[y], chunk.x_world, y * SECTION_SIZE, chunk.z_world)
        self.chunk = chunk

    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (section_x, section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (section_x, section_y, section_z).
        """
        return self.data.transpose((2, 0, 1))

    def get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_section()
