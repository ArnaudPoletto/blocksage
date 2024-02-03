import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import torch
import numpy as np
from typing import Tuple
from torch.nn.functional import one_hot

from torch.utils.data import Dataset, DataLoader

from src.utils.log import log
from src.utils.block_dictionary import get_block_id_dictionary
from src.config import (
    SKIPGRAM_WINDOW_SIZE, 
    CLUSTER_SIZE, 
    SECTION_SIZE,
    CLUSTER_DATASET_PATH,
    TRAIN_SPLIT,
    TEST_SPLIT,
    VAL_SPLIT,
    BATCH_SIZE,
    NUM_WORKERS,
)


class SkipGramDataset(Dataset):
    """A skip-gram dataset."""

    def __init__(
        self,
        cluster_file_paths: str,
        window_size: int = SKIPGRAM_WINDOW_SIZE,
        block_id_dict: dict = None,
    ):
        """
        Initialize the skip-gram dataset.

        Args:
            cluster_file_paths: List of cluster file paths.
            block_id_dict: Block id dictionary. Defaults to None.
        """
        super(SkipGramDataset, self).__init__()

        # Get block id dictionary if not provided
        if block_id_dict is None:
            block_id_dict = get_block_id_dictionary()

        self.cluster_file_paths = cluster_file_paths
        self.window_size = window_size
        self.vocabulary_size = len(block_id_dict) + 1  # +1 for the masked block
        # The number of blocks per cluster is the number of valid blocks, not considering the blocks
        # in the borders of the cluster. The number of neighbors is the number of blocks in the window,
        # not considering the center block.
        self.valid_cluster_size = CLUSTER_SIZE * SECTION_SIZE - (2 * self.window_size)
        self.total_window_size = 2 * self.window_size + 1
        self.n_neighbors = self.total_window_size**3 - 1

    def __len__(self) -> int:
        """
        Get the length of the skip-gram dataset.

        Returns:
            int: Length of the skip-gram dataset.
        """
        return (
            len(self.cluster_file_paths)
            * (self.valid_cluster_size**3)
            * self.n_neighbors
        )

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
        Get an skip-gram sample from the skip-gram dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            torch.Tensor: Item from the skip-gram dataset.
        """
        # Get indices
        n_entries_per_cluster = (self.valid_cluster_size**3) * self.n_neighbors
        cluster_file_path_idx = idx // n_entries_per_cluster
        entry_idx = idx % n_entries_per_cluster
        block_idx = entry_idx // self.n_neighbors
        block_x_idx = block_idx // (self.valid_cluster_size**2)
        block_y_idx = (
            block_idx % (self.valid_cluster_size**2)
        ) // self.valid_cluster_size
        block_z_idx = (
            block_idx % (self.valid_cluster_size**2)
        ) % self.valid_cluster_size
        neighbor_idx = entry_idx % self.n_neighbors

        # # Get cluster
        # cluster_file_path = self.cluster_file_paths[cluster_file_path_idx]
        # cluster = torch.from_numpy(np.load(cluster_file_path))
        # cluster = SkipGramDataset._reshape_to_3d(cluster)

        # # Get target block
        # target_block_id = cluster[
        #     block_x_idx + self.window_size,
        #     block_y_idx + self.window_size,
        #     block_z_idx + self.window_size,
        # ]

        # # Get positive context block
        # zone_around_block = cluster[
        #     block_x_idx - self.window_size : block_x_idx + self.window_size + 1,
        #     block_y_idx - self.window_size : block_y_idx + self.window_size + 1,
        #     block_z_idx - self.window_size : block_z_idx + self.window_size + 1,
        # ].flatten() # First, get the zone around the target block
        # center_block_idx = (self.total_window_size**3 // 2) + 1
        # neighbors = torch.cat(
        #     (
        #         zone_around_block[:center_block_idx],
        #         zone_around_block[center_block_idx + 1 :],
        #     )
        # ).flatten() # Remove the target block from the zone
        # positive_context_block_id = neighbors[neighbor_idx]

        # # Get negative context block
        # unique_neighbors = set(torch.unique(neighbors).tolist())
        # negative_values = list(set(range(self.vocabulary_size)) - unique_neighbors)
        # negative_context_block_id = np.random.choice(negative_values)

        # # Get as one-hot vectors
        # target_block = one_hot(
        #     torch.tensor(target_block_id), num_classes=self.vocabulary_size
        # )
        # positive_context_block = one_hot(
        #     torch.tensor(positive_context_block_id), num_classes=self.vocabulary_size
        # )
        # negative_context_block = one_hot(
        #     torch.tensor(negative_context_block_id), num_classes=self.vocabulary_size
        # )

        return 1
        #return target_block, positive_context_block, negative_context_block
    

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
        raise ValueError(
            f"❌ Subset fraction must be in (0, 1]. Got {subset_fraction} instead."
        )

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
        log(
            f"Using a subset of the whole cluster dataset with {n_cluster_file_paths_subset} clusters ({subset_fraction * 100}%)."
        )

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
    window_size: int = SKIPGRAM_WINDOW_SIZE,
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
        window_size (int, optional): Window size. Defaults to SKIPGRAM_WINDOW_SIZE.
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
    if block_id_dict is None:
        block_id_dict = get_block_id_dictionary()
    train_dataset = SkipGramDataset(
        train_cluster_file_paths,
        block_id_dict = block_id_dict,
        window_size = window_size,
    )
    test_dataset = SkipGramDataset(
        test_cluster_file_paths,
        block_id_dict = block_id_dict,
        window_size = window_size,
    )
    val_dataset = SkipGramDataset(
        val_cluster_file_paths,
        block_id_dict = block_id_dict,
        window_size = window_size,
    )

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
