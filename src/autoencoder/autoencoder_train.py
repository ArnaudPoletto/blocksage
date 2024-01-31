import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import torch.nn as nn
from typing import List

from src.utils.log import log
from src.utils.random import set_seed
from src.utils.parser import get_config
from src.models.autoencoder import AutoEncoder
from src.utils.block_dictionary import get_block_id_dictionary
from src.trainers.autoencoder_trainer import AutoEncoderTrainer
from src.utils.segmentation_train import get_criterion, get_optimizer
from src.config import (
    DATA_PATH,
    ENCODER_CONV_CHANNELS,
    DECODER_CONV_CHANNELS,
    DEVICE,
    SEED,
    DATASET_SUBSET_FRACTION,
)
from src.datasets.cluster_dataset import get_dataloaders

AUTOENCODER_PATH = f"{DATA_PATH}models/autoencoder/"
AUTOENCODER_NAME = "autoencoder"


def get_model(
    input_size: int,
    output_size: int,
    encoder_conv_channels: List[int] = ENCODER_CONV_CHANNELS,
    decoder_conv_channels: List[int] = DECODER_CONV_CHANNELS,
    dropout_rate: float = 0.0,
    with_pooling: bool = True,
) -> nn.Module:
    """
    Get the model.

    Args:
        input_size (int): Input size.
        output_size (int): Output size.
        encoder_conv_channels (List[int], optional): List of encoder convolutional channels. Defaults to ENCODER_CONV_CHANNELS.
        decoder_conv_channels (List[int], optional): List of decoder convolutional channels. Defaults to DECODER_CONV_CHANNELS.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        with_pooling (bool, optional): Whether to use pooling. Defaults to True.

    Returns:
        nn.Module: The model
    """
    autoencoder = AutoEncoder(
        input_size = input_size,
        output_size = output_size,
        encoder_conv_channels = encoder_conv_channels,
        decoder_conv_channels = decoder_conv_channels,
        dropout_rate = dropout_rate,
        with_pooling = with_pooling,
    )

    learnable_parameters = sum(
        p.numel() for p in autoencoder.parameters() if p.requires_grad
    )
    log(f"🔢 Number of learnable parameters: {learnable_parameters:,}")

    autoencoder = autoencoder.to(DEVICE)
    return autoencoder


def get_trainer(
    model: nn.Module,
    criterion: nn.Module,
    accumulation_steps: int,
    evaluation_steps: int,
    use_scaler: bool,
) -> AutoEncoderTrainer:
    """
    Get the trainer.

    Args:
        model (nn.Module): Model.
        criterion (nn.Module): Criterion.
        accumulation_steps (int): Accumulation steps.
        evaluation_steps (int): Evaluation steps.
        use_scaler (bool): Whether to use scaler.

    Returns:
        SiameseUNetConcTrainer: The trainer
    """
    return AutoEncoderTrainer(
        model = model,
        criterion = criterion,
        accumulation_steps = accumulation_steps,
        evaluation_steps = evaluation_steps,
        print_statistics = False,
        use_scaler = use_scaler,
        name = AUTOENCODER_NAME,
    )


if __name__ == "__main__":
    set_seed(SEED)

    # Get parameters from config file
    config = get_config(f"{GLOBAL_DIR}/config/autoencoder_best_params.yml")
    encoder_conv_channels = config["encoder_conv_channels"]
    decoder_conv_channels = config["decoder_conv_channels"]
    dropout_rate = config["dropout_rate"]
    criterion_name = config["criterion_name"]
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    accumulation_steps = int(config["accumulation_steps"])
    evaluation_steps = int(config["evaluation_steps"])
    use_scaler = bool(config["use_scaler"])
    num_epochs = int(config["num_epochs"])
    with_pooling = bool(config["with_pooling"])

    # Get input and output sizes
    block_id_dict = get_block_id_dictionary()
    input_size = len(block_id_dict) + 1 # Add 1 for masked blocks
    output_size = input_size

    # Get dataloaders, model, criterion, optimizer, and trainer
    train_loader, test_loader, val_loader = get_dataloaders(
        block_id_dict=block_id_dict,
        subset_fraction=DATASET_SUBSET_FRACTION
        )
    model = get_model(
        input_size = input_size,
        output_size = output_size,
        encoder_conv_channels = encoder_conv_channels,
        decoder_conv_channels = decoder_conv_channels,
        dropout_rate = dropout_rate,
        with_pooling = with_pooling,
    )
    criterion = get_criterion(
        criterion_name = criterion_name, 
        block_id_dict = block_id_dict
    )
    optimizer = get_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    trainer = get_trainer(
        model = model,
        criterion = criterion,
        accumulation_steps = accumulation_steps,
        evaluation_steps = evaluation_steps,
        use_scaler = use_scaler,
    )

    # Train the model
    statistics = trainer.train(
        train_loader = train_loader,
        val_loader = val_loader,
        optimizer = optimizer,
        num_epochs = num_epochs,
        learning_rate = learning_rate,
    )
