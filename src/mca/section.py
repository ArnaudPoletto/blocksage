import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np

from src.mca.zone import Zone
from src.utils.log import warn
from src.config import SECTION_SIZE


class Section(Zone):
    """A section of a chunk. There are 24 sections in a chunk."""

    SHAPE_SIZE = 3

    def __init__(
        self, data: np.ndarray, x_world: int, y_world: int, z_world: int
    ) -> None:
        """
        Initialize a section.

        Args:
            data (np.ndarray): The array containing the block indices.
            x_world (int): The x coordinate of the section in the world.
            y_world (int): The y coordinate of the section in the world.
            z_world (int): The z coordinate of the section in the world.
        """
        if len(data.shape) != self.SHAPE_SIZE:
            raise ValueError(
                f"❌ section_blocks must be of shape (section_x, section_y, section_z), not {data.shape}."
            )
        if (
            data.shape[0] != SECTION_SIZE
            or data.shape[1] != SECTION_SIZE
            or data.shape[2] != SECTION_SIZE
        ):
            warn(
                f"The section data do not fit the expected shape (section_x, section_y, section_z) = ({SECTION_SIZE}, {SECTION_SIZE}, {SECTION_SIZE}), got {data.shape} instead."
            )
            
        
        super().__init__(data, x_world, y_world, z_world)

    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (section_x, section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (section_x, section_y, section_z).
        """
        return self.data.transpose((2, 0, 1))

    def get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_section()
