import numpy as np

from src.mca.zone import Zone
from src.utils.log import warn
from src.mca.section import Section
from src.utils.block_dictionary import get_block_id_dictionary
from src.config import (
    AIR_NAME,
    SECTION_SIZE,
    NATURAL_UNDERGROUND_BLOCK_NAMES,
    MAX_N_SECTIONS_PER_CLUSTER_PER_DIM,
    CLUSTER_SIZE,
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
        cluster_size: int = CLUSTER_SIZE,
    ) -> None:
        """
        Initialize a cluster.

        Args:
            data (np.ndarray): Array containing the block indices of shape (cluster_x, cluster_y, cluster_z, section_y, section_z, section_x).
            x_world (int, optional): x coordinate of the cluster in the world. Defaults to 0.
            y_world (int, optional): y coordinate of the cluster in the world. Defaults to 0.
            z_world (int, optional): z coordinate of the cluster in the world. Defaults to 0.
            cluster_size (int, optional): Number of sections per cluster per dimension. Defaults to CLUSTER_SIZE.

        Raises:
            ValueError: If the data do not have the expected shape.
            ValueError: If the cluster size is out of bounds, or even.
        """
        if len(data.shape) != self.SHAPE_SIZE:
            raise ValueError(
                f"❌ cluster_blocks must be of shape (cluster_x, cluster_y, cluster_z, section_y, section_z, section_x), not {data.shape}."
            )
        if (
            data.shape[0] != cluster_size
            or data.shape[1] != cluster_size
            or data.shape[2] != cluster_size
            or data.shape[3] != SECTION_SIZE
            or data.shape[4] != SECTION_SIZE
            or data.shape[5] != SECTION_SIZE
        ):
            warn(
                f"The region data do not fit the expected shape (cluster_x, cluster_y, cluster_z, section_y, section_z, section_x) = ({cluster_size}, {cluster_size}, {cluster_size}, {SECTION_SIZE}, {SECTION_SIZE}, {SECTION_SIZE}), got {data.shape} instead."
            )
        if cluster_size < 0 or cluster_size > MAX_N_SECTIONS_PER_CLUSTER_PER_DIM:
            raise ValueError(
                f"❌ cluster_size must be in [0, {MAX_N_SECTIONS_PER_CLUSTER_PER_DIM}], not {cluster_size}."
            )
        if cluster_size % 2 == 0:
            raise ValueError(f"❌ cluster_size must be odd, not {cluster_size}.")

        super().__init__(data, x_world, y_world, z_world)
        self.cluster_size = cluster_size

    def get_data_for_display(self) -> np.ndarray:
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

    def is_relevant(self, block_id_dict: dict = None) -> bool:
        """
        Check whether the cluster is relevant. A cluster is relevant if the following requirements are met:
        - There is no non-loaded section in the cluster.
        - The center section has at most 90% of non-air blocks.
        - The center section has at most 90% of air blocks.
        - The cluster has less than 70% of natural underground blocks.

        Args:
            block_id_dict (dict, optional): Dictionary of block IDs. Defaults to None.

        Returns:
            bool: Whether the cluster is relevant.
        """
        if block_id_dict is None:
            block_id_dict = get_block_id_dictionary()

        # No non-loaded section in the cluster
        if np.any(self.data == np.uint16(-1)):
            return False

        # Non-air blocks percentage at most 90% in the center section
        center_section = self.get_section(
            self.cluster_size // 2, self.cluster_size // 2, self.cluster_size // 2
        )
        center_section_data = center_section.get_data_by_section()
        center_section_n_blocks = center_section_data.size

        air_id = block_id_dict[AIR_NAME]
        n_non_air_blocks = np.count_nonzero(center_section_data != air_id)
        percent_non_air_blocks = n_non_air_blocks / center_section_n_blocks

        if percent_non_air_blocks > 0.9:
            return False

        # Air blocks percentage at most 90% in the center section
        n_air_blocks = np.count_nonzero(center_section_data == air_id)
        percent_air_blocks = n_air_blocks / center_section_n_blocks

        if percent_air_blocks > 0.9:
            return False

        # More than 70% of natural underground blocks in the cluster
        natural_underground_block_ids = [
            block_id_dict[name]
            for name in block_id_dict
            if name in NATURAL_UNDERGROUND_BLOCK_NAMES
        ]
        n_natural_underground_blocks = np.count_nonzero(
            np.isin(self.data, natural_underground_block_ids)
        )
        percent_natural_underground_blocks = (
            n_natural_underground_blocks / self.data.size
        )

        if percent_natural_underground_blocks > 0.7:
            return False

        return True
