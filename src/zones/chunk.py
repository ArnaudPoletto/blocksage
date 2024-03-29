import numpy as np

from src.utils.log import warn
from src.zones.zone import Zone
from src.zones.section import Section
from src.config import (
    MIN_Y,
    SECTION_SIZE,
    N_SECTIONS_PER_CLUSTER_PER_DIM,
)


class Chunk(Zone):
    """A chunk of a region. There are 32x32 chunks in a region."""

    SHAPE_SIZE = 4

    def __init__(self, data: np.array, x_world: int, z_world: int) -> None:
        """
        Initialize a region.

        Args:
            data (np.ndarray): Array containing the block indices of shape (section, section_y, section_z, section_x).
            x_world (int, optional): x coordinate of the region in the world.
            z_world (int, optional): z coordinate of the region in the world.

        Raises:
            ValueError: If the data do not have the expected shape.
        """
        if len(data.shape) != self.SHAPE_SIZE:
            raise ValueError(
                f"❌ chunk_blocks must be of shape (section, section_y, section_z, section_x), not {data.shape}."
            )
        if (
            data.shape[0] != N_SECTIONS_PER_CLUSTER_PER_DIM
            or data.shape[1] != SECTION_SIZE
            or data.shape[2] != SECTION_SIZE
            or data.shape[3] != SECTION_SIZE
        ):
            warn(
                f"The region data do not fit the expected shape (section, section_y, section_z, section_x) = ({N_SECTIONS_PER_CLUSTER_PER_DIM}, {SECTION_SIZE}, {SECTION_SIZE}, {SECTION_SIZE}), got {data.shape} instead."
            )

        super().__init__(data, x_world, MIN_Y, z_world)

    def _get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_chunk()

    def get_section(self, y: int) -> Section:
        """
        Get a section of blocks.

        Args:
            y (int): y coordinate of the section.

        Raises:
            ValueError: If the y coordinate is out of bounds.

        Returns:
            Section: Section of blocks.
        """
        if y < 0 or y >= self.data.shape[0]:
            raise ValueError(f"❌ y must be in [0, {self.data.shape[0]}), not {y}.")

        return Section(self.data[y], self.x_world, y * SECTION_SIZE, self.z_world)

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
