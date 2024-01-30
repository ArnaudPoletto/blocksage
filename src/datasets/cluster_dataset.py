import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot

from src.config import SECTION_SIZE


class ClusterDataset(Dataset):
    """A cluster dataset."""

    @staticmethod
    def _load_cluster_dataset(
        cluster_dataset: str,
        ) -> list:
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

        # Shuffle the cluster file paths
        np.random.shuffle(cluster_file_paths)

        return cluster_file_paths

    def __init__(self, cluster_dataset_path: str, num_block_classes: int) -> None:
        """
        Initialize a cluster dataset.

        Args:
            cluster_dataset_path (str): The path to the cluster dataset.
            num_block_classes (int): The number of block classes.
        """

        self.cluster_file_paths = ClusterDataset._load_cluster_dataset(
            cluster_dataset_path
        )
        self.num_block_classes = num_block_classes
        self.num_total_classes = num_block_classes + 1 # Add 1 for masked block

    def __len__(self) -> int:
        """
        Get the length of the cluster dataset.

        Returns:
            int: The length of the cluster dataset.
        """
        return len(self.cluster_file_paths)
    
    @staticmethod
    def reshape_to_3d(cluster: torch.Tensor) -> torch.Tensor:
        """
        Reshape a cluster to a 3d tensor.

        Args:
            cluster (torch.Tensor): The cluster.

        Returns:
            torch.Tensor: The reshaped cluster.
        """
        region_x, region_z, section, section_y, section_z, section_x = cluster.shape
        return cluster \
            .transpose((0, 5, 1, 4, 2, 3)) \
            .reshape((region_x * section_x, region_z * section_z, section * section_y)) \
            .transpose((0, 2, 1))

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a cluster from the cluster dataset.

        Args:
            idx (int): The index of the cluster.

        Returns:
            torch.Tensor: The cluster.
        """
        # Get cluster file and load it
        cluster_file_path = self.cluster_file_paths[idx]
        cluster_gt = np.load(cluster_file_path)

        # Mask the center section
        cluster_masked = cluster_gt.copy()
        cluster_masked[
            cluster_masked.shape[0] // 2,
            cluster_masked.shape[1] // 2,
            cluster_masked.shape[2] // 2,
        ] = self.num_block_classes # Masked block are represented by the last index

        # Reshape tensors as 3d tensors
        cluster_masked = ClusterDataset.reshape_to_3d(cluster_masked)
        cluster_gt = ClusterDataset.reshape_to_3d(cluster_gt)

        # # Only take the center section for the ground truth
        # cluster_size = cluster_gt.shape[2] // SECTION_SIZE
        # center_section_start = SECTION_SIZE * (cluster_size // 2)
        # center_section_end = center_section_start + SECTION_SIZE
        # cluster_gt = cluster_gt[
        #     center_section_start:center_section_end,
        #     center_section_start:center_section_end,
        #     center_section_start:center_section_end,
        # ]

        # To tensor
        cluster_masked = torch.from_numpy(cluster_masked)
        cluster_gt = torch.from_numpy(cluster_gt)

        # To one hot tensors
        cluster_masked = one_hot(cluster_masked.long(), self.num_total_classes) \
            .float() \
            .permute(3, 0, 1, 2)
        cluster_gt = one_hot(cluster_gt.long(), self.num_total_classes).float() \
            .permute(3, 0, 1, 2)

        return cluster_masked, cluster_gt
    
def get_cluster_dataloader(
    cluster_dataset_path: str,
    num_block_classes: int,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Get a cluster dataloader.

    Args:
        cluster_dataset_path (str): The path to the cluster dataset.
        num_block_classes (int): The number of block classes.
        batch_size (int, optional): The batch size. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        num_workers (int, optional): The number of workers. Defaults to 0.

    Returns:
        DataLoader: The cluster dataloader.
    """
    cluster_dataset = ClusterDataset(cluster_dataset_path, num_block_classes)
    return DataLoader(
        cluster_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
