from typing import Any
from torch.utils.data import DataLoader

from src.models.skipgram import SkipGram
from src.skipgram.skipgram_train import get_optimizer
from src.trainers.skipgram_trainer import SkipGramTrainer
from src.config import (
    DEVICE,
    SWEEP_NUM_EVALUATIONS_PER_EPOCH,
)


class SkipGramSweeper:
    """
    Sweeper class for sweeping the parameter space of a skipgram model.
    """

    def __init__(
        self,
        config: Any,
        vocabulary_size: int,
    ):
        """
        Initialize the skipgram sweeper.

        Args:
            config (Any): Wandb config.
            vocabulary_size (int): Vocabulary size.

        """
        self.config = config
        self.vocabulary_size = vocabulary_size

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Train the model.

        Args:
            train_loader (DataLoader): Train loader.
            val_loader (DataLoader): Validation loader.
        """
        # Get parameters from config
        embedding_dimension = self.config.embedding_dimension
        learning_rate = self.config.learning_rate
        weight_decay = self.config.weight_decay
        accumulation_steps = self.config.accumulation_steps
        use_scaler = self.config.use_scaler
        num_epochs = self.config.num_epochs
        evaluation_steps = int(len(train_loader) // SWEEP_NUM_EVALUATIONS_PER_EPOCH)

        # Get model, optimizer and trainer
        self.model = SkipGram(
            vocabulary_size=self.vocabulary_size,
            embedding_dimension=embedding_dimension,
        ).to(DEVICE)

        self.optimizer = get_optimizer(
            model=self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.trainer = SkipGramTrainer(
            model=self.model,
            criterion=None,  # TODO: add criterion for other models
            accumulation_steps=accumulation_steps,
            evaluation_steps=evaluation_steps,
            use_scaler=use_scaler,
        )

        self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=self.optimizer,
            num_epochs=num_epochs,
            save_model=False,
            sweeping=True,
        )
