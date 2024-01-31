import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import torch
import numpy as np
from typing import Tuple
from src.utils.log import log
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot

from src.utils.block_dictionary import get_block_id_dictionary
from src.config import (
    CLUSTER_DATASET_PATH,
    TRAIN_SPLIT,
    TEST_SPLIT,
    VAL_SPLIT,
    BATCH_SIZE,
    NUM_WORKERS,
)


class ClusterDataset(Dataset):
    """A cluster dataset."""

    def __init__(
        self,
        cluster_file_paths: str,
        num_block_classes: int,
    ) -> None:
        """
        Initialize a cluster dataset.

        Args:
            cluster_file_paths (List[str]): List of cluster file paths.
            num_block_classes (int): Number of block classes.

        """

        self.cluster_file_paths = cluster_file_paths
        self.num_block_classes = num_block_classes
        self.num_total_classes = num_block_classes + 1  # Add 1 for masked block

    def __len__(self) -> int:
        """
        Get the length of the cluster dataset.

        Returns:
            int: Length of the cluster dataset.
        """
        return len(self.cluster_file_paths)

    @staticmethod
    def _reshape_to_3d(cluster: torch.Tensor) -> torch.Tensor:
        """
        Reshape a cluster as an array of shape (region_x * section_x, region_z * section_z, section * section_y).

        Args:
            cluster (torch.Tensor): Array representing a cluster of shape (region_x, region_z, section, section_y, section_z, section_x).

        Returns:
            torch.Tensor: Array representing a cluster of shape (region_x * section_x, region_z * section_z, section * section_y).
        """
        region_x, region_z, section, section_y, section_z, section_x = cluster.shape
        return (
            cluster.transpose((0, 5, 1, 4, 2, 3))
            .reshape((region_x * section_x, region_z * section_z, section * section_y))
            .transpose((0, 2, 1))
        )

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

        # # Mask the center section
        cluster_in = cluster_gt.copy()
        # cluster_masked[
        #     cluster_masked.shape[0] // 2,
        #     cluster_masked.shape[1] // 2,
        #     cluster_masked.shape[2] // 2,
        # ] = self.num_block_classes  # Masked block are represented by the last index

        # Reshape tensors as 3d tensors
        cluster_in = ClusterDataset._reshape_to_3d(cluster_in)
        cluster_gt = ClusterDataset._reshape_to_3d(cluster_gt)

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
        cluster_in = torch.from_numpy(cluster_in)
        cluster_gt = torch.from_numpy(cluster_gt)

        # To one hot tensors
        cluster_in = (
            one_hot(cluster_in.long(), self.num_total_classes)
            .float()
            .permute(3, 0, 1, 2)
        )
        cluster_gt = (
            one_hot(cluster_gt.long(), self.num_total_classes)
            .float()
            .permute(3, 0, 1, 2)
        )

        return cluster_in, cluster_gt


def _load_cluster_file_paths(
        cluster_dataset: str,
        subset_fraction: float = 1.0,
    ) -> list:
    """
    Load the cluster dataset.

    Args:
        cluster_dataset (str): Path to the cluster dataset.
        subset_fraction (float, optional): Fraction of the dataset to use. Defaults to 1.0.

    Raises:
        ValueError: If subset fraction is not in (0, 1].

    Returns:
        list: List of cluster file paths.
    """
    if subset_fraction <= 0 or subset_fraction > 1:
        raise ValueError(f"❌ Subset fraction must be in (0, 1]. Got {subset_fraction} instead.")
    
    cluster_file_paths = []
    for cluster_folder in os.listdir(cluster_dataset):
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

    # Get subset of cluster file paths
    n_cluster_file_paths = len(cluster_file_paths)
    n_cluster_file_paths_subset = int(n_cluster_file_paths * subset_fraction)
    cluster_file_paths = cluster_file_paths[:n_cluster_file_paths_subset]
    if subset_fraction < 1.0:
        log(f"Using a subset of the whole cluster dataset with {n_cluster_file_paths_subset} clusters ({subset_fraction * 100}%).")

    return cluster_file_paths


def _split_cluster_file_paths(
    cluster_file_paths: list,
    train_split: float = TRAIN_SPLIT,
    test_split: float = TEST_SPLIT,
    val_split: float = VAL_SPLIT,
) -> Tuple[list, list, list]:
    """
    Split the cluster file paths.

    Args:
        cluster_file_paths (list): List of cluster file paths.
        train_split (float, optional): Train split. Defaults to TRAIN_SPLIT.
        test_split (float, optional): Test split. Defaults to TEST_SPLIT.
        val_split (float, optional): Validation split. Defaults to VAL_SPLIT.

    Returns:
        list: List of train, test and validation cluster file paths.
    """
    if np.abs(train_split + test_split + val_split - 1) > 1e-3:
        raise ValueError(
            f"Train, test and validation split must sum to 1. Got {train_split + test_split + val_split} instead."
        )

    # Shuffle the cluster file paths
    np.random.shuffle(cluster_file_paths)

    # Split the cluster file paths
    n_cluster_file_paths = len(cluster_file_paths)
    train_split_idx = int(n_cluster_file_paths * train_split)
    test_split_idx = int(n_cluster_file_paths * (train_split + test_split))

    train_cluster_file_paths = cluster_file_paths[:train_split_idx]
    test_cluster_file_paths = cluster_file_paths[train_split_idx:test_split_idx]
    val_cluster_file_paths = cluster_file_paths[test_split_idx:]

    return train_cluster_file_paths, test_cluster_file_paths, val_cluster_file_paths


def get_dataloaders(
    cluster_dataset: str = CLUSTER_DATASET_PATH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    train_split: float = TRAIN_SPLIT,
    test_split: float = TEST_SPLIT,
    val_split: float = VAL_SPLIT,
    block_id_dict: dict = None,
    subset_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get the dataloaders.

    Args:
        cluster_dataset (str, optional): Path to the cluster dataset. Defaults to CLUSTER_DATASET_PATH.
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.
        num_workers (int, optional): Number of workers. Defaults to NUM_WORKERS.
        train_split (float, optional): Train split. Defaults to TRAIN_SPLIT.
        test_split (float, optional): Test split. Defaults to TEST_SPLIT.
        val_split (float, optional): Validation split. Defaults to VAL_SPLIT.
        block_id_dict (dict, optional): Block id dictionary. Defaults to None.
        subset_fraction (float, optional): Fraction of the dataset to use. Defaults to 1.0.


    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Tuple of train, test and validation dataloaders.
    """
    if np.abs(train_split + test_split + val_split - 1) > 1e-3:
        raise ValueError(
            f"Train, test and validation split must sum to 1. Got {train_split + test_split + val_split} instead."
        )

    # Get block id dictionary if not provided
    if block_id_dict is None:
        block_id_dict = get_block_id_dictionary()

    # Load cluster file paths
    cluster_file_paths = _load_cluster_file_paths(cluster_dataset, subset_fraction)

    # Split cluster file paths
    (
        train_cluster_file_paths,
        test_cluster_file_paths,
        val_cluster_file_paths,
    ) = _split_cluster_file_paths(
        cluster_file_paths,
        train_split=train_split,
        test_split=test_split,
        val_split=val_split,
    )

    # Get datasets
    num_block_classes = len(block_id_dict)
    train_dataset = ClusterDataset(train_cluster_file_paths, num_block_classes=num_block_classes)
    test_dataset = ClusterDataset(test_cluster_file_paths, num_block_classes=num_block_classes)
    val_dataset = ClusterDataset(val_cluster_file_paths, num_block_classes=num_block_classes)

    # Get dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )

    return train_dataloader, test_dataloader, val_dataloader
