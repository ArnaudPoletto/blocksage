import numpy as np
from typing import Dict

from src.utils.log import warn
from src.zones.zone import Zone
from src.zones.section import Section
from src.utils.block_dictionary import get_block_id_dictionary
from src.config import (
    AIR_NAME,
    CLUSTER_SIZE,
    SECTION_SIZE,
    MASKED_BLOCK_ID,
    N_SECTIONS_PER_CLUSTER_PER_DIM,
)


class Cluster(Zone):
    """A cluster of sections. There are 9 clusters in a cluster, in a 3x3 grid."""

    SHAPE_SIZE = 6

    def __init__(
        self,
        data: np.ndarray,
        x_world: int = 0,
        y_world: int = 0,
        z_world: int = 0,
    ) -> None:
        """
        Initialize a cluster.

        Args:
            data (np.ndarray): Array containing the block indices of shape (cluster_x, cluster_y, cluster_z, section_y, section_z, section_x).
            x_world (int, optional): x coordinate of the cluster in the world. Defaults to 0.
            y_world (int, optional): y coordinate of the cluster in the world. Defaults to 0.
            z_world (int, optional): z coordinate of the cluster in the world. Defaults to 0.

        Raises:
            ValueError: If the data do not have the expected shape.
        """
        if len(data.shape) != self.SHAPE_SIZE:
            raise ValueError(
                f"❌ cluster_blocks must be of shape (cluster_x, cluster_y, cluster_z, section_y, section_z, section_x), not {data.shape}."
            )
        if (
            data.shape[0] != CLUSTER_SIZE
            or data.shape[1] != CLUSTER_SIZE
            or data.shape[2] != CLUSTER_SIZE
            or data.shape[3] != SECTION_SIZE
            or data.shape[4] != SECTION_SIZE
            or data.shape[5] != SECTION_SIZE
        ):
            warn(
                f"The region data do not fit the expected shape (cluster_x, cluster_y, cluster_z, section_y, section_z, section_x) = ({CLUSTER_SIZE}, {CLUSTER_SIZE}, {CLUSTER_SIZE}, {SECTION_SIZE}, {SECTION_SIZE}, {SECTION_SIZE}), got {data.shape} instead."
            )

        super().__init__(data, x_world, y_world, z_world)

    def _get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_cluster()

    def get_section(self, x: int, y: int, z: int) -> Section:
        """
        Get a section of blocks.

        Args:
            x (int): x coordinate of the section in the cluster.
            y (int): y coordinate of the section in the cluster.
            z (int): z coordinate of the section in the cluster.

        Raises:
            ValueError: If the x, y or z coordinates are out of bounds.

        Returns:
            np.ndarray: The section of blocks.
        """
        if x < 0 or x >= self.data.shape[0]:
            raise ValueError(f"❌ x must be in [0, {self.data.shape[0]}), not {x}.")
        if y < 0 or y >= self.data.shape[1]:
            raise ValueError(f"❌ y must be in [0, {self.data.shape[1]}), not {y}.")
        if z < 0 or z >= self.data.shape[2]:
            raise ValueError(f"❌ z must be in [0, {self.data.shape[2]}), not {z}.")

        data = self.data[x, y, z]
        x_world = self.x_world + x * SECTION_SIZE
        y_world = self.y_world + y * SECTION_SIZE
        z_world = self.z_world + z * SECTION_SIZE
        return Section(data, x_world, y_world, z_world)

    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (cluster_x, cluster_y, cluster_z, section_x, section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (cluster_x, cluster_y, cluster_z, section_x, section_y, section_z).
        """
        return self.data.transpose((0, 1, 2, 5, 3, 4))

    def get_data_by_cluster(self) -> np.ndarray:
        """
        View the blocks by cluster, i.e. as an array of shape (cluster_x, cluster_y, cluster_z).

        Returns:
            np.ndarray: Array of block IDs of shape (cluster_x, cluster_y, cluster_z).
        """
        region_x, region_z, section, section_y, section_z, section_x = self.data.shape

        return (
            self.data.transpose((0, 5, 1, 4, 2, 3))
            .reshape((region_x * section_x, region_z * section_z, section * section_y))
            .transpose((0, 2, 1))
        )

    def is_relevant(
            self, 
            block_id_dict: Dict[str, int],
            threshold_percent_air_blocks: float = 0.9
            ) -> bool:
        """
        Check whether the cluster is relevant. A cluster is relevant if the following requirements are met:
        - There is no non-loaded section in the cluster.
        - The center section has at most 100*threshold_percent_air_blocks% of air blocks.

        Args:
            block_id_dict (Dict[str, int]): Dictionary of block IDs. Defaults to None.
            threshold_percent_air_blocks (float, optional): Threshold percentage of air blocks in the center section. Defaults to 0.9.

        Returns:
            bool: Whether the cluster is relevant.
        """
        # No non-loaded section in the cluster
        if np.any(self.data == MASKED_BLOCK_ID):
            return False

        # Non-air blocks percentage at most 90% in the center section
        center_section = self.get_section(
            CLUSTER_SIZE // 2, CLUSTER_SIZE // 2, CLUSTER_SIZE // 2
        )
        center_section_data = center_section.get_data_by_section()
        center_section_n_blocks = center_section_data.size
        air_id = block_id_dict[AIR_NAME]
        
        # Air blocks percentage at most 90% in the center section
        n_air_blocks = np.count_nonzero(center_section_data == air_id)
        percent_air_blocks = n_air_blocks / center_section_n_blocks

        if percent_air_blocks > threshold_percent_air_blocks:
            return False

        return True
