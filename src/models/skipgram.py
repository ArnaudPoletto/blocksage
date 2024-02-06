# Adapted from: https://github.com/fanglanting/skip-gram-pytorch/tree/master

import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

from src.utils.log import log


class SkipGram(nn.Module):
    """Skip-gram model."""

    def _initialize_embeddings(self) -> None:
        """
        Initialize the embeddings.
        """
        initialization_range = 0.5 / self.embedding_dimension
        self.target_embeddings.weight.data.uniform_(
            -initialization_range, initialization_range
        )
        self.context_embeddings.weight.data.uniform_(-0, 0)

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
    ) -> None:
        """
        Initialize the skip-gram model.

        Args:
            vocabulary_size (int): Vocabulary size.
            embedding_dimension (int): Embedding dimension.
        """
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension

        self.target_embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.context_embeddings = nn.Embedding(vocabulary_size, embedding_dimension)

        self._initialize_embeddings()

    def forward(
        self,
        target_block: torch.Tensor,
        positive_context_block: torch.Tensor,
        negative_context_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            target_block (torch.Tensor): Representation of the target block as block id. The tensor has shape (batch_size).
            positive_context_block (torch.Tensor): Representation of the positive context block as block id. The tensor has shape (batch_size).
            negative_context_blocks (torch.Tensor): Representation of the set of negative context blocks as block id. The tensor has shape (batch_size, num_negative_samples).

        Returns:
            torch.Tensor: Loss, defined as the sum of positive and negative log likelihoods.
        """
        batch_size = target_block.size(0)

        # Compute embeddings
        target_block_embeddings = self.target_embeddings(target_block)
        positive_context_block_embeddings = self.context_embeddings(
            positive_context_block
        )
        negative_context_blocks_embeddings = self.context_embeddings(
            negative_context_blocks
        )

        # Get positive score
        positive_score = torch.mul(
            target_block_embeddings, positive_context_block_embeddings
        )
        positive_score = torch.sum(positive_score, dim=1)
        positive_target = F.logsigmoid(positive_score).squeeze()

        # Get negative score
        negative_score = torch.bmm(
            negative_context_blocks_embeddings, target_block_embeddings.unsqueeze(2)
        ).squeeze()
        negative_score = torch.sum(negative_score, dim=1)
        negative_target = F.logsigmoid(
            -negative_score
        ).squeeze()  # Negative samples have negative labels

        # Compute loss as the sum of positive and negative log likelihoods
        loss = positive_target + negative_target
        loss = -loss.sum() / batch_size

        return loss

    def get_input_embeddings(self) -> torch.Tensor:
        """
        Get input embeddings.

        Returns:
            torch.Tensor: Input embeddings.
        """
        return self.target_embeddings.weight.data.cpu().numpy()

    def save_input_embeddings(self, path: str) -> None:
        """
        Save input embeddings.

        Args:
            path (str): Path to save input embeddings.
        """
        np.save(path, self.get_input_embeddings())
        log(f"💾 Saved input embeddings to {Path(path).resolve()}")

    def save_model(self, path: str) -> None:
        """
        Save the model.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)
        log(f"💾 Saved model to {Path(path).resolve()}")
