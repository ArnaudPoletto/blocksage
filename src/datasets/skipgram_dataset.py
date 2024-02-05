import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
import numpy as np
from torch.nn.functional import one_hot

from torch.utils.data import Dataset, DataLoader

from src.config import (
    SKIPGRAM_COOCCURRENCE_MATRIX_PATH,
    SKIPGRAM_NUM_WORKERS,
)


class SkipGramDataset(Dataset):
    """A skip-gram dataset."""

    def __init__(
        self,
        cooccurrence_matrix: np.ndarray,
        dataset_size: int,
        num_negative_samples: int,
    ):
        """
        Initialize the skip-gram dataset.

        Args:
            cooccurrence_matrix (np.ndarray): Co-occurrence matrix.
            dataset_size (int): Size of the dataset.
            num_negative_samples (int, optional): Number of negative samples to draw.

        Raises:
            ValueError: If the co-occurrence matrix is not a square matrix.
            ValueError: If the dataset size is not positive.
            ValueError: If the number of negative samples is not positive.
        """
        super().__init__()

        if (
            len(cooccurrence_matrix.shape) != 2
            or cooccurrence_matrix.shape[0] != cooccurrence_matrix.shape[1]
        ):
            raise ValueError(
                f"❌ Co-occurrence matrix must be a square matrix. Got shape {cooccurrence_matrix.shape} instead."
            )

        if dataset_size <= 0:
            raise ValueError(
                f"❌ Dataset size must be positive. Got {dataset_size} instead."
            )

        if num_negative_samples <= 0:
            raise ValueError(
                f"❌ Number of negative samples must be positive. Got {num_negative_samples} instead."
            )

        self.cooccurrence_matrix = cooccurrence_matrix
        self.dataset_size = dataset_size
        self.num_negative_samples = num_negative_samples

    def __len__(self) -> int:
        """
        Get the length of the skip-gram dataset.

        Returns:
            int: Length of the skip-gram dataset.
        """
        return self.dataset_size

    def __getitem__(self, idx: int):
        vocabulary_size = self.cooccurrence_matrix.shape[0]

        # Pick random target block
        target_block = np.random.randint(vocabulary_size)

        # Pick random positive context block given the probability distribution
        positive_distribution = self.cooccurrence_matrix[target_block]
        positive_context_block = np.random.choice(
            vocabulary_size,
            p=positive_distribution
        )

        # Pick negative context block uniformly at random, excluding the positive context block
        negative_distribution = np.ones(vocabulary_size) / (vocabulary_size - 1)
        negative_distribution[target_block] = 0
        negative_context_blocks = np.random.choice(
            vocabulary_size,
            p=negative_distribution,
            size=self.num_negative_samples,
        )

        # To tensors
        target_block = torch.tensor(target_block)
        positive_context_block = torch.tensor(positive_context_block)
        negative_context_blocks = torch.tensor(negative_context_blocks)

        return target_block, positive_context_block, negative_context_blocks


def get_dataloader(
    dataset_size: int,
    num_negative_samples: int,
    batch_size: int,
    num_workers: int = SKIPGRAM_NUM_WORKERS,
) -> DataLoader:
    """
    Get the dataloader for the skip-gram dataset.

    Args:
        dataset_size (int, optional): Size of the dataset.
        num_negative_samples (int, optional): Number of negative samples to draw.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers. Defaults to SKIPGRAM_NUM_WORKERS.

    Returns:
        DataLoader: The dataloader for the skip-gram dataset.
    """
    # Get cooccurrence matrix
    cooccurrence_matrix = np.load(SKIPGRAM_COOCCURRENCE_MATRIX_PATH)

    # Get dataset
    dataset = SkipGramDataset(
        cooccurrence_matrix=cooccurrence_matrix,
        dataset_size=dataset_size,
        num_negative_samples=num_negative_samples,
    )

    # Get dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    return dataloader
