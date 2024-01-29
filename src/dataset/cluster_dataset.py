import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from src.config import DATASET_MASKED_BLOCK_ID, SECTION_SIZE


class ClusterDataset(Dataset):
    """A cluster dataset."""

    @staticmethod
    def _load_cluster_dataset(cluster_dataset: str) -> list:
        """
        Load the cluster dataset.

        Args:
            cluster_dataset (str): The path to the cluster dataset.

        Returns:
            list: A list of cluster file paths.
        """
        cluster_file_paths = []
        for cluster_folder in tqdm(
            os.listdir(cluster_dataset), desc="🔄 Loading clusters"
        ):
            cluster_folder_path = os.path.join(cluster_dataset, cluster_folder)
            if not os.path.isdir(cluster_folder_path):
                continue

            # Iterate through each region file in the folder
            for cluster_file in os.listdir(cluster_folder_path):
                cluster_file_path = os.path.join(cluster_folder_path, cluster_file)

                # Add cluster file path to cluster paths
                cluster_file_paths.append(cluster_file_path)

        return cluster_file_paths

    def __init__(self, cluster_dataset_path: str) -> None:
        """
        Initialize a cluster dataset.

        Args:
            cluster_dataset_path (str): The path to the cluster dataset.
        """

        self.cluster_file_paths = ClusterDataset._load_cluster_dataset(
            cluster_dataset_path
        )

    def __len__(self) -> int:
        """
        Get the length of the cluster dataset.

        Returns:
            int: The length of the cluster dataset.
        """
        return len(self.cluster_file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a cluster from the cluster dataset.

        Args:
            idx (int): The index of the cluster.

        Returns:
            torch.Tensor: The cluster.
        """
        cluster_file_path = self.cluster_file_paths[idx]
        cluster_gt = torch.from_numpy(np.load(cluster_file_path)).float()

        # Mask the center section
        cluster_masked = cluster_gt.clone()
        center_section_x = cluster_masked.shape[0] // 2
        center_section_y = cluster_masked.shape[1] // 2
        center_section_z = cluster_masked.shape[2] // 2
        cluster_masked[
            center_section_x : center_section_x + SECTION_SIZE,
            center_section_y : center_section_y + SECTION_SIZE,
            center_section_z : center_section_z + SECTION_SIZE,
        ] = DATASET_MASKED_BLOCK_ID

        return cluster_masked, cluster_gt
