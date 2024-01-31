import torch
from torch import nn
from typing import override
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from src.trainers.trainer import Trainer

from src.config import DEVICE

class AutoEncoderTrainer(Trainer):
    """
    Trainer class used to train an autoencoder model.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        accumulation_steps: int,
        evaluation_steps: int,
        print_statistics: bool = False,
        use_scaler: bool = False,
        name: str = "",
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model (nn.Module): Model to train.
            criterion (nn.Module): Loss function to use.
            accumulation_steps (int): Accumulation steps for gradient accumulation.
            evaluation_steps (int): Evaluation steps for evaluation.
            print_statistics (bool, optional): Whether to print statistics during training. Defaults to False.
            use_scaler (bool, optional): Whether to use scaler. Defaults to False.
            name (str, optional): Name of the model. Defaults to the empty string.
        """
        super().__init__(
            model=model,
            criterion=criterion,
            accumulation_steps=accumulation_steps,
            evaluation_steps=evaluation_steps,
            print_statistics=print_statistics,
            use_scaler=use_scaler,
            name=name,
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
        name += f"_{self.criterion.__class__.__name__}loss"
        name += f"_{self.model.activation.__class__.__name__}act"
        name += f"_{str(self.model.dropout_rate).replace('.', '')}dropout"
        name += f"_{self.accumulation_steps}accsteps"
        name += f"_{self.evaluation_steps}evalsteps"
        name += f"_{len(self.model.encoder_conv_channels)}encconvs"
        name += f"_{len(self.model.decoder_conv_channels)}decconvs"
        name += f"_{'' if self.model.with_pooling else 'no'}pool"

        return name
    
    def _forward_pass(self, batch: tuple) -> torch.Tensor:
        # Unpack batch
        cluster_masked, cluster_gt = batch
        cluster_masked = cluster_masked.to(DEVICE)
        cluster_gt = cluster_gt.to(DEVICE)

        # Forward pass
        with autocast(enabled=self.use_scaler):
            outputs = self.model(cluster_masked)

        # Get predictions and targets
        pred = torch.argmax(outputs, dim=1)
        targets = torch.argmax(cluster_gt, dim=1)

        # Compute loss
        loss = self.criterion(outputs, cluster_gt)

        return loss, pred, targets