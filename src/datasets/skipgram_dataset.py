import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data import Dataset, DataLoader

from src.config import (
    SKIPGRAM_NUM_WORKERS,
    SKIPGRAM_COOCCURRENCE_MATRIX_PATH,
    SKIPGRAM_UNIGRAM_DISTRIBUTION_PATH,
)


class SkipGramDataset(Dataset):
    """A skip-gram dataset."""

    def __init__(
        self,
        cooccurrence_matrix: np.ndarray,
        dataset_size: int,
        num_negative_samples: int,
        unigram_distribution: np.ndarray,
        noise_power: float = 0.75,
        subsampling_threshold: float = 1e-3,
        eps: float = 1e-5,
    ):
        """
        Initialize the skip-gram dataset.

        Args:
            cooccurrence_matrix (np.ndarray): Co-occurrence matrix.
            dataset_size (int): Size of the dataset.
            num_negative_samples (int, optional): Number of negative samples to draw.
            unigram_distribution (np.ndarray): The probability distribution of individual blocks in the training corpus.
            noise_power (float, optional): The exponent used to shape the noise distribution. Defaults to 0.75.
            subsampling_threshold (float, optional): The threshold used to subsample frequent words. Defaults to 1e-3.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.

        Raises:
            ValueError: If the co-occurrence matrix is not a square matrix.
            ValueError: If the co-occurrence matrix rows do not sum to 1.
            ValueError: If the dataset size is not positive.
            ValueError: If the number of negative samples is not positive.
            ValueError: If the unigram distribution does not sum to 1.
            ValueError: If the unigram distribution is not between 0 and 1.
        """
        super().__init__()

        if (
            len(cooccurrence_matrix.shape) != 2
            or cooccurrence_matrix.shape[0] != cooccurrence_matrix.shape[1]
        ):
            raise ValueError(
                f"❌ Co-occurrence matrix must be a square matrix. Got shape {cooccurrence_matrix.shape} instead."
            )

        if not np.all(np.isclose(np.sum(cooccurrence_matrix, axis=1), 1)):
            raise ValueError(
                f"❌ Co-occurrence matrix rows must sum to 1. Got {np.sum(cooccurrence_matrix, axis=1)} instead."
            )

        if dataset_size <= 0:
            raise ValueError(
                f"❌ Dataset size must be positive. Got {dataset_size} instead."
            )

        if num_negative_samples <= 0:
            raise ValueError(
                f"❌ Number of negative samples must be positive. Got {num_negative_samples} instead."
            )

        if not np.isclose(unigram_distribution.sum(), 1):
            raise ValueError(
                f"❌ Unigram distribution must sum to 1. Got {unigram_distribution.sum()} instead."
            )

        if np.any(unigram_distribution < 0) or np.any(unigram_distribution > 1):
            raise ValueError(
                f"❌ Unigram distribution must be between 0 and 1. Got {unigram_distribution} instead."
            )

        self.dataset_size = dataset_size
        self.num_negative_samples = num_negative_samples
        # Noise distribution to sample negative context blocks
        self.noise_distribution = unigram_distribution**noise_power
        # Subsampling distribution to subsample frequent words
        # See https://naturale0.github.io/2021/02/08/understanding-skip-gram#subsampling
        subsampling_fraction = subsampling_threshold / (unigram_distribution + eps)
        subsampling_distribution = np.sqrt(subsampling_fraction) + subsampling_fraction
        # subsampling_distribution = np.clip(subsampling_distribution, 0, 1)
        self.subsampled_cooccurrence_matrix = (
            cooccurrence_matrix * subsampling_distribution
        )
        self.subsampled_cooccurrence_matrix /= self.subsampled_cooccurrence_matrix.sum(
            axis=1, keepdims=True
        )

    def __len__(self) -> int:
        """
        Get the length of the skip-gram dataset.

        Returns:
            int: Length of the skip-gram dataset.
        """
        return self.dataset_size

    def __getitem__(self, idx: int):
        vocabulary_size = self.subsampled_cooccurrence_matrix.shape[0]

        # Pick random target block
        target_block = np.random.choice(vocabulary_size)

        # Pick random positive context block given the probability distribution
        positive_distribution = self.subsampled_cooccurrence_matrix[target_block]
        positive_context_block = np.random.choice(
            vocabulary_size, p=positive_distribution
        )

        # Pick negative context block uniformly at random, excluding the positive context block
        noise_distribution = self.noise_distribution.copy()
        noise_distribution[positive_context_block] = 0
        noise_distribution /= noise_distribution.sum()
        negative_context_blocks = np.random.choice(
            vocabulary_size,
            p=noise_distribution,
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
    # Get cooccurrence matrix and unigram distribution
    cooccurrence_matrix = np.load(SKIPGRAM_COOCCURRENCE_MATRIX_PATH)
    uni_gram_distribution = np.load(SKIPGRAM_UNIGRAM_DISTRIBUTION_PATH)

    # Get dataset
    dataset = SkipGramDataset(
        cooccurrence_matrix=cooccurrence_matrix,
        dataset_size=dataset_size,
        num_negative_samples=num_negative_samples,
        unigram_distribution=uni_gram_distribution,
    )

    # Get dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    return dataloader
