import torch
from torch import nn
from typing import Tuple
from typing import override
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from src.trainers.trainer import Trainer
from src.models.skipgram import SkipGram
from src.config import DEVICE


class SkipGramTrainer(Trainer):
    """
    Trainer class used to train a skip-gram model.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        accumulation_steps: int,
        evaluation_steps: int,
        name: str,
        use_scheduler: bool,
        use_scaler: bool,
        print_statistics: bool,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model (nn.Module): Model to train.
            criterion (nn.Module): Loss function to use.
            accumulation_steps (int): Accumulation steps for gradient accumulation.
            evaluation_steps (int): Evaluation steps for evaluation.
            name (str, optional): Name of the model. Defaults to the empty string.
            use_scheduler (bool, optional): Whether to use a learning rate scheduler.
            use_scaler (bool, optional): Whether to use scaler.
            print_statistics (bool, optional): Whether to print statistics during training.

        Raises:
            ValueError: If the model is not an instance of SkipGram.
        """
        if not isinstance(model, SkipGram):
            raise ValueError("❌ Model must be an instance of SkipGram.")

        super().__init__(
            model=model,
            criterion=criterion,
            accumulation_steps=accumulation_steps,
            evaluation_steps=evaluation_steps,
            name=name,
            use_scheduler=use_scheduler,
            use_scaler=use_scaler,
            print_statistics=print_statistics,
        )

    @override
    def _get_name(
        self, optimizer: Optimizer, num_epochs: int, learning_rate: float
    ) -> str:
        name = self.name
        name += self.model.__class__.__name__
        name += f"_{optimizer.__class__.__name__}optim"
        name += f"_{num_epochs}epochs"
        name += f"_{str(learning_rate).replace('.', '')}lr"
        name += f"_{self.model.vocabulary_size}vocsize"
        name += f"_{self.model.embedding_dimension}embdim"

        return name

    def _forward_pass(self, batch: tuple) -> Tuple[torch.Tensor, None, None]:
        # Unpack batch
        target_block, positive_context_block, negative_context_blocks = batch
        target_block = target_block.to(DEVICE)
        positive_context_block = positive_context_block.to(DEVICE)
        negative_context_blocks = negative_context_blocks.to(DEVICE)

        # Forward pass and get loss
        with autocast(enabled=self.use_scaler):
            loss = self.model(
                target_block, positive_context_block, negative_context_blocks
            )

        return loss, None, None
