
import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np

from src.mca.zone import Zone
from src.mca.section import Section
from src.utils.block_dictionary import get_block_id_dictionary
from src.config import (
        DEFAULT_N_SECTIONS_PER_CLUSTER_PER_DIM, 
        MAX_N_SECTIONS_PER_CLUSTER_PER_DIM, 
        SECTION_SIZE, 
        AIR_NAME,
        NATURAL_UNDERGROUND_BLOCK_NAMES
    )

class Cluster(Zone):
    """A cluster of sections. There are 9 clusters in a cluster, in a 3x3 grid."""

    def __init__(self, region, x: int, y: int, z: int, cluster_size: int = DEFAULT_N_SECTIONS_PER_CLUSTER_PER_DIM) -> None:
        """
        Initialize a cluster.

        Args:
            region (Region): The region the cluster belongs to.
            x (int): The x coordinate of the cluster.
            y (int): The y coordinate of the cluster.
            z (int): The z coordinate of the cluster.
            cluster_size (int, optional): The number of sections per cluster per dimension. Defaults to DEFAULT_N_SECTIONS_PER_CLUSTER_PER_DIM.
        """
        if cluster_size < 0 or cluster_size > MAX_N_SECTIONS_PER_CLUSTER_PER_DIM:
            raise ValueError(f"❌ cluster_size must be in [0, {MAX_N_SECTIONS_PER_CLUSTER_PER_DIM}], not {cluster_size}.")
        if cluster_size % 2 == 0:
            raise ValueError(f"❌ cluster_size must be odd, not {cluster_size}.")
        x_max = region.data.shape[0] - cluster_size + 1
        z_max = region.data.shape[1] - cluster_size + 1
        y_max = region.data.shape[2] - cluster_size + 1
        if x < 0 or x >= x_max:
            raise ValueError(f"❌ x must be in [0, {x_max}), not {x}.")
        if z < 0 or z >= z_max:
            raise ValueError(f"❌ z must be in [0, {z_max}), not {z}.")
        if y < 0 or y >= y_max:
            raise ValueError(f"❌ y must be in [0, {y_max}), not {y}.")

        cluster_data = region.data[x:x + cluster_size, z:z + cluster_size, y:y + cluster_size]
        cluster_x_world = region.x_world + x * SECTION_SIZE
        cluster_y_world = region.y_world + y * SECTION_SIZE
        cluster_z_world = region.z_world + z * SECTION_SIZE
        super().__init__(cluster_data, cluster_x_world, cluster_y_world, cluster_z_world)
        self.region = region
        self.x = x
        self.y = y
        self.z = z
        self.cluster_size = cluster_size

    def get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_cluster()
    
    def get_section(self, x: int, y: int, z: int) -> Section:
        """
        Returns a section of blocks.

        Args:
            x (int): The x coordinate of the section in the cluster.
            y (int): The y coordinate of the section in the cluster.
            z (int): The z coordinate of the section in the cluster.

        Returns:
            np.ndarray: The section of blocks.
        """
        if x < 0 or x >= self.data.shape[0]:
            raise ValueError(f"❌ x must be in [0, {self.data.shape[0]}), not {x}.")
        if y < 0 or y >= self.data.shape[1]:
            raise ValueError(f"❌ y must be in [0, {self.data.shape[1]}), not {y}.")
        if z < 0 or z >= self.data.shape[2]:
            raise ValueError(f"❌ z must be in [0, {self.data.shape[2]}), not {z}.")

        chunk = self.region.get_chunk(self.x + x, self.z + z)
        return Section(chunk, self.y + y)
    
    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (cluster_size, cluster_size, cluster_size, section_x, section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (cluster_size, 3cluster_size, cluster_size, section_x, section_y, section_z).
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
            self.data.transpose(0, 5, 1, 4, 2, 3)
            .reshape((region_x * section_x, region_z * section_z, section * section_y))
            .transpose(0, 2, 1)
        )
    
    def is_relevant(self, block_id_dict: dict = None) -> bool:
        """
        A cluster is relevant if the following requirements are met:
        - The center section has at most 90% of non-air blocks.
        - The center section has at most 90% of air blocks.
        - The cluster has less than 70% of natural underground blocks.

        Args:
            block_id_dict (dict): The dictionary of block IDs. Defaults to None.

        Returns:
            bool: Whether the cluster is relevant.
        """
        if block_id_dict is None:
            block_id_dict = get_block_id_dictionary()

        # Non-air blocks percentage at most 90% in the center section
        center_section = self.get_section(self.cluster_size // 2, self.cluster_size // 2, self.cluster_size // 2)
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
        natural_underground_block_ids = [block_id_dict[name] for name in block_id_dict if name in NATURAL_UNDERGROUND_BLOCK_NAMES]
        n_natural_underground_blocks = np.count_nonzero(np.isin(self.data, natural_underground_block_ids))
        percent_natural_underground_blocks = n_natural_underground_blocks / self.data.size

        if percent_natural_underground_blocks > 0.7:
            return False
        
        return True

        
