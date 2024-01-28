
import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np

from src.mca.zone import Zone
from src.config import N_SECTIONS_PER_CLUSTER_PER_DIM, SECTION_SIZE

class Cluster(Zone):
    """A cluster of sections. There are 9 clusters in a cluster, in a 3x3 grid."""

    def __init__(self, region, x: int, y: int, z: int) -> None:
        """
        Initialize a cluster.

        Args:
            region (Region): The region the cluster belongs to.
            x (int): The x coordinate of the cluster.
            z (int): The z coordinate of the cluster.
        """
        if x < 0 or x >= region.data.shape[0] - N_SECTIONS_PER_CLUSTER_PER_DIM:
            raise ValueError(f"❌ x must be in [0, {region.data.shape[0] - N_SECTIONS_PER_CLUSTER_PER_DIM}), not {x}.")
        if z < 0 or z >= region.data.shape[1] - N_SECTIONS_PER_CLUSTER_PER_DIM:
            raise ValueError(f"❌ z must be in [0, {region.data.shape[1] - N_SECTIONS_PER_CLUSTER_PER_DIM}), not {z}.")
        if y < 0 or y >= region.data.shape[2] - N_SECTIONS_PER_CLUSTER_PER_DIM:
            raise ValueError(f"❌ y must be in [0, {region.data.shape[1] - N_SECTIONS_PER_CLUSTER_PER_DIM}), not {y}.")

        cluster_data = region.data[x:x + N_SECTIONS_PER_CLUSTER_PER_DIM, z:z + N_SECTIONS_PER_CLUSTER_PER_DIM, y:y + N_SECTIONS_PER_CLUSTER_PER_DIM]
        cluster_x_world = region.x_world + x * SECTION_SIZE
        cluster_y_world = region.y_world + y * SECTION_SIZE
        cluster_z_world = region.z_world + z * SECTION_SIZE
        super().__init__(cluster_data, cluster_x_world, cluster_y_world, cluster_z_world)
        self.region = region

    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (3, 3, 3, section_x, section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (3, 3, 3, section_x, section_y, section_z).
        """
        return self.data.transpose((0, 1, 2, 5, 3, 4))
    
    def get_data_by_chunk(self) -> np.ndarray:
        """
        View the blocks by chunk, i.e. as an array of shape (3, 3, chunk_x, chunk_y, chunk_z) = (3, 3, section_x, section * section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (3, 3, chunk_x, chunk_y, chunk_z) = (3, 3, section_x, section * section_y, section_z).
        """
        region_x, region_z, section, section_y, section_z, section_x = self.data.shape

        return self.data \
            .reshape((region_x, region_z, section * section_y, section_z, section_x)) \
            .transpose((0, 1, 4, 2, 3))

    def get_data_by_cluster(self) -> np.ndarray:
        """
        View the blocks by cluster, i.e. as an array of shape (cluster_x, cluster_y, cluster_z).

        Returns:
            np.ndarray: Array of block IDs of shape (cluster_x, cluster_y, cluster_z).
        """
        region_x, region_z, section, section_y, section_z, section_x = self.data.shape

        return (
            self.data.transpose(0, 5, 1, 4, 2, 3)
            .reshape((region_x * section_x, region_z * section_z, section * section_y))
            .transpose(0, 2, 1)
        )

    def get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_cluster()