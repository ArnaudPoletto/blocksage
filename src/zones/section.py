import numpy as np

from src.zones.zone import Zone
from src.utils.log import warn
from src.config import SECTION_SIZE


class Section(Zone):
    """A section of a chunk. There are 24 sections in a chunk."""

    SHAPE_SIZE = 3

    def __init__(
        self, data: np.ndarray, x_world: int = 0, y_world: int = 0, z_world: int = 0
    ) -> None:
        """
        Initialize a section.

        Args:
            data (np.ndarray): Array containing the block indices of shape (section_y, section_z, section_x).
            x_world (int, optional): x coordinate of the section in the world.
            y_world (int, optional): y coordinate of the section in the world.
            z_world (int, optional): z coordinate of the section in the world.

        Raises:
            ValueError: If the data do not have the expected shape.
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

    def _get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_section()

    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (section_x, section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (section_x, section_y, section_z).
        """
        return self.data.transpose((2, 0, 1))
