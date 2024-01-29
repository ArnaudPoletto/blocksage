import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np
from typing import Generator

from src.mca.zone import Zone
from src.mca.chunk import Chunk
from src.mca.cluster import Cluster
from src.mca.section import Section
from src.utils.log import warn
from src.config import (
    MIN_Y,
    SECTION_SIZE,
    CHUNK_XZ_SIZE,
    DEFAULT_CLUSTER_STRIDE,
    MAX_N_SECTIONS_PER_CLUSTER_PER_DIM,
    DEFAULT_N_SECTIONS_PER_CLUSTER_PER_DIM,
    N_CHUNKS_PER_REGION_PER_DIM,
)


class Region(Zone):
    """A region as a collection of blocks of shape (region_x, region_z, section, section_y, section_z, section_x)."""

    SHAPE_SIZE = 6

    def __init__(self, data: np.ndarray, x_world: int = 0, z_world: int = 0) -> None:
        """
        Initialize a region.

        Args:
            data (np.ndarray): The array containing the block indices of shape (region_x, region_z, section, section_y, section_z, section_x).
            x_world (int): The x coordinate of the region in the world. Defaults to 0.
            z_world (int): The z coordinate of the region in the world. Defaults to 0.
        """
        if len(data.shape) != self.SHAPE_SIZE:
            raise ValueError(
                f"❌ region_blocks must be of shape (region_x, region_z, section, section_y, section_z, section_x), not {data.shape}."
            )
        if (
            data.shape[0] != N_CHUNKS_PER_REGION_PER_DIM
            or data.shape[1] != N_CHUNKS_PER_REGION_PER_DIM
            or data.shape[2] != MAX_N_SECTIONS_PER_CLUSTER_PER_DIM
            or data.shape[3] != SECTION_SIZE
            or data.shape[4] != SECTION_SIZE
            or data.shape[5] != SECTION_SIZE
        ):
            warn(
                f"The region data do not fit the expected shape (region_x, region_z, section, section_y, section_z, section_x) = ({N_CHUNKS_PER_REGION_PER_DIM}, {N_CHUNKS_PER_REGION_PER_DIM}, {MAX_N_SECTIONS_PER_CLUSTER_PER_DIM}, {SECTION_SIZE}, {SECTION_SIZE}, {SECTION_SIZE}), got {data.shape} instead."
            )

        super().__init__(data, x_world, MIN_Y, z_world)

    def get_data_for_display(self) -> np.ndarray:
        return self.get_data_by_region()

    def get_chunk(self, x: int, z: int) -> Chunk:
        """
        Returns a chunk of blocks.

        Args:
            x (int): The x coordinate of the chunk.
            z (int): The z coordinate of the chunk.

        Returns:
            np.ndarray: The chunk of blocks.
        """
        if x < 0 or x >= self.data.shape[0]:
            raise ValueError(f"❌ x must be in [0, {self.data.shape[0]}), not {x}.")
        if z < 0 or z >= self.data.shape[1]:
            raise ValueError(f"❌ z must be in [0, {self.data.shape[1]}), not {z}.")

        x_world = self.x_world + CHUNK_XZ_SIZE * x
        z_world = self.z_world + CHUNK_XZ_SIZE * z
        return Chunk(self.data[x, z], x_world, z_world)

    def get_cluster(
        self,
        x: int,
        y: int,
        z: int,
        cluster_size: int = DEFAULT_N_SECTIONS_PER_CLUSTER_PER_DIM,
    ) -> Cluster:
        """
        Returns a cluster of blocks.

        Args:
            x (int): The x coordinate of the cluster.
            y (int): The y coordinate of the cluster.
            z (int): The z coordinate of the cluster.
            cluster_size (int, optional): The number of sections per cluster per dimension. Defaults to DEFAULT_N_SECTIONS_PER_CLUSTER_PER_DIM.

        Returns:
            np.ndarray: The cluster of blocks.
        """
        if cluster_size < 0 or cluster_size > MAX_N_SECTIONS_PER_CLUSTER_PER_DIM:
            raise ValueError(
                f"❌ cluster_size must be in [0, {MAX_N_SECTIONS_PER_CLUSTER_PER_DIM}], not {cluster_size}."
            )
        if cluster_size % 2 == 0:
            raise ValueError(f"❌ cluster_size must be odd, not {cluster_size}.")
        x_max = self.data.shape[0] - cluster_size + 1
        z_max = self.data.shape[1] - cluster_size + 1
        y_max = self.data.shape[2] - cluster_size + 1
        if x < 0 or x >= x_max:
            raise ValueError(f"❌ x must be in [0, {x_max}), not {x}.")
        if z < 0 or z >= z_max:
            raise ValueError(f"❌ z must be in [0, {z_max}), not {z}.")
        if y < 0 or y >= y_max:
            raise ValueError(f"❌ y must be in [0, {y_max}), not {y}.")

        data = self.data[
            x : x + cluster_size, z : z + cluster_size, y : y + cluster_size
        ]
        x_world = self.x_world + x * SECTION_SIZE
        y_world = self.y_world + y * SECTION_SIZE
        z_world = self.z_world + z * SECTION_SIZE
        return Cluster(data, x_world, y_world, z_world, cluster_size)

    def get_section(self, x: int, y: int, z: int) -> Section:
        """
        Returns a section of blocks.

        Args:
            x (int): The x coordinate of the section.
            y (int): The y coordinate of the section.
            z (int): The z coordinate of the section.

        Returns:
            np.ndarray: The section of blocks.
        """
        if x < 0 or x >= self.data.shape[0]:
            raise ValueError(f"❌ x must be in [0, {self.data.shape[0]}), not {x}.")
        if z < 0 or z >= self.data.shape[1]:
            raise ValueError(f"❌ z must be in [0, {self.data.shape[1]}), not {z}.")
        if y < 0 or y >= self.data.shape[2]:
            raise ValueError(f"❌ y must be in [0, {self.data.shape[2]}), not {y}.")

        return self.get_chunk(x, z).get_section(y)

    def get_data_by_region(self) -> np.ndarray:
        """
        View the blocks by region, i.e. as an array of shape (region_x, region_y, region_z) = (region_x * section_x, section * section_y, region_z * section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (region_x, region_y, region_z) = (region_x * section_x, section * section_y, region_z * section_z).
        """
        region_x, region_z, section, section_y, section_z, section_x = self.data.shape

        return (
            self.data.transpose((0, 5, 1, 4, 2, 3))
            .reshape((region_x * section_x, region_z * section_z, section * section_y))
            .transpose((0, 2, 1))
        )

    def get_data_by_chunk(self) -> np.ndarray:
        """
        View the blocks by chunk, i.e. as an array of shape (region_x, region_z, chunk_x, chunk_y, chunk_z) = (region_x, region_z, section_x, section * section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (region_x, region_z, chunk_x, chunk_y, chunk_z) = (region_x, region_z, section_x, section * section_y, section_z).
        """
        region_x, region_z, section, section_y, section_z, section_x = self.data.shape

        return self.data.reshape(
            (region_x, region_z, section * section_y, section_z, section_x)
        ).transpose((0, 1, 4, 2, 3))

    def get_data_by_section(self) -> np.ndarray:
        """
        View the blocks by section, i.e. as an array of shape (region_x, region_z, section, section_x, section_y, section_z).

        Returns:
            np.ndarray: Array of block IDs of shape (region_x, region_z, section, section_x, section_y, section_z).
        """
        return self.data.transpose((0, 1, 2, 5, 3, 4))

    def get_clusters(
        self,
        block_id_dict: dict = None,
        cluster_size: int = DEFAULT_N_SECTIONS_PER_CLUSTER_PER_DIM,
        stride: int = DEFAULT_CLUSTER_STRIDE,
        only_relevant: bool = True,
    ) -> Generator[Cluster, None, None]:
        """
        Returns a generator of clusters of blocks.

        Args:
            block_id_dict (dict, optional): The dictionary of block IDs. Defaults to None.
            cluster_size (int): The number of sections per cluster per dimension.
            stride (int): The stride between clusters.
            only_relevant (bool): Whether to only return relevant clusters.

        Returns:
            Generator[Cluster]: The clusters of blocks.
        """
        if cluster_size < 0 or cluster_size > MAX_N_SECTIONS_PER_CLUSTER_PER_DIM:
            raise ValueError(
                f"❌ cluster_size must be in [0, {MAX_N_SECTIONS_PER_CLUSTER_PER_DIM}], not {cluster_size}."
            )
        if cluster_size % 2 == 0:
            raise ValueError(f"❌ cluster_size must be odd, not {cluster_size}.")
        if stride < 1 or stride > cluster_size:
            raise ValueError(f"❌ stride must be in [1, {cluster_size}], not {stride}.")

        region_x, region_z, section, _, _, _ = self.data.shape
        for x in range(0, region_x - cluster_size + 1, stride):
            for z in range(0, region_z - cluster_size + 1, stride):
                for y in range(0, section - cluster_size + 1, stride):
                    data = self.data[
                        x : x + cluster_size, z : z + cluster_size, y : y + cluster_size
                    ]
                    x_world = self.x_world + x * SECTION_SIZE
                    y_world = self.y_world + y * SECTION_SIZE
                    z_world = self.z_world + z * SECTION_SIZE
                    cluster = Cluster(data, x_world, y_world, z_world, cluster_size)
                    
                    if only_relevant and not cluster.is_relevant(block_id_dict):
                        continue

                    yield cluster
