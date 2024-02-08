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
        noise_power: float,
        minimum_noise_distribution: float,
        subsampling_threshold: float,
        minimum_subsampling_distribution: float,
    ):
        """
        Initialize the skip-gram dataset.

        Args:
            cooccurrence_matrix (np.ndarray): Co-occurrence matrix.
            dataset_size (int): Size of the dataset.
            num_negative_samples (int): Number of negative samples to draw.
            unigram_distribution (np.ndarray): The probability distribution of individual blocks in the training corpus.
            noise_power (float): The exponent used to shape the noise distribution.
            minimum_noise_distribution (float): The minimum distribution a noise block can have.
            subsampling_threshold (float): The threshold used to subsample frequent words.
            minimum_subsampling_distribution (float): The minimum distribution a subsampled block can have.

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

        if not np.all(
            np.logical_or(
                np.isclose(np.sum(cooccurrence_matrix, axis=1), 0),
                np.isclose(np.sum(cooccurrence_matrix, axis=1), 1),
            )
        ):
            raise ValueError(
                f"❌ Co-occurrence matrix rows must either sum up to 1 for blocks found in the corpus or 0 for blocks not found in the corpus. Got {np.sum(cooccurrence_matrix, axis=1)} instead."
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
        # See https://naturale0.github.io/2021/02/08/understanding-skip-gram#negative-sampling
        # Modified to avoid not sampling blocks not found in the corpus, using a minimum noise distribution
        noise_distribution = np.maximum(
            unigram_distribution**noise_power, minimum_noise_distribution
        )
        noise_distribution /= noise_distribution.sum()
        self.noise_distribution = noise_distribution
        # Subsampling distribution to subsample frequent words
        # See https://naturale0.github.io/2021/02/08/understanding-skip-gram#subsampling
        # Modified to avoid getting more than a factor of 2 in the subsampling distribution
        subsampling_fraction = subsampling_threshold / (
            unigram_distribution + subsampling_threshold
        )
        subsampling_distribution = np.sqrt(subsampling_fraction) + subsampling_fraction
        self.subsampled_cooccurrence_matrix = np.maximum(
            cooccurrence_matrix * subsampling_distribution, minimum_subsampling_distribution
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
    noise_power: float,
    minimum_noise_distribution: float,
    subsampling_threshold: float,
    minimum_subsampling_distribution: float,
) -> DataLoader:
    """
    Get the dataloader for the skip-gram dataset.

    Args:
        dataset_size (int): Size of the dataset.
        num_negative_samples (int): Number of negative samples to draw.
        batch_size (int): Batch size.
        noise_power (float): The exponent used to shape the noise distribution.
        minimum_noise_distribution (float): The minimum distribution a noise block can have.
        subsampling_threshold (float): The threshold used to subsample frequent words.
        minimum_subsampling_distribution (float): The minimum distribution a subsampled block can have.

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
        noise_power=noise_power,
        minimum_noise_distribution=minimum_noise_distribution,
        subsampling_threshold=subsampling_threshold,
        minimum_subsampling_distribution=minimum_subsampling_distribution,
    )

    # Get dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=SKIPGRAM_NUM_WORKERS, shuffle=True
    )

    return dataloader
