import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import torch.nn as nn
from torch.optim import AdamW

from src.utils.log import log
from src.utils.random import set_seed
from src.utils.parser import get_config
from src.models.skipgram import SkipGram
from src.trainers.skipgram_trainer import SkipGramTrainer
from src.utils.block_dictionary import get_block_id_dictionary
from src.datasets.skipgram_dataset import get_dataloader
from src.config import (
    DEVICE,
    SEED,
    SKIPGRAM_NAME,
    SKIPGRAM_CONFIG_PATH,
    SKIPGRAM_TRAIN_DATASET_SIZE,
    SKIPGRAM_VAL_DATASET_SIZE,
    SKIPGRAM_EMBEDDINGS_PATH,
    SKIPGRAM_MODEL_PATH,
)


def _get_model(
    vocabulary_size: int,
    embedding_dimension: int,
) -> nn.Module:
    """
    Get the model.

    Args:
        vocabulary_size (int): Vocabulary size.
        embedding_dimension (int): Embedding dimension.

    Returns:
        nn.Module: The model
    """
    model = SkipGram(
        vocabulary_size=vocabulary_size,
        embedding_dimension=embedding_dimension,
    )

    learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"🔢 Number of learnable parameters: {learnable_parameters:,}")

    model = model.to(DEVICE)
    return model


def _get_trainer(
    model: nn.Module,
    accumulation_steps: int,
    evaluation_steps: int,
    use_scaler: bool,
) -> SkipGramTrainer:
    """
    Get the trainer.

    Args:
        model (nn.Module): Model.
        accumulation_steps (int): Accumulation steps.
        evaluation_steps (int): Evaluation steps.
        use_scaler (bool): Whether to use scaler.

    Returns:
        SkipGramTrainer: The trainer
    """
    return SkipGramTrainer(
        model=model,
        criterion=None,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        print_statistics=False,
        use_scaler=use_scaler,
        name=SKIPGRAM_NAME,
    )


def get_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
):
    """
    Get the optimizer.

    Args:
        model (nn.Module): Model.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.

    Returns:
        torch.optim.Optimizer: The optimizer
    """
    return AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


if __name__ == "__main__":
    set_seed(SEED)

    # Get parameters from config file
    config = get_config(SKIPGRAM_CONFIG_PATH)
    num_negative_samples = int(config["num_negative_samples"])
    batch_size = int(config["batch_size"])
    embedding_dimension = int(config["embedding_dimension"])
    accumulation_steps = int(config["accumulation_steps"])
    evaluation_steps = int(config["evaluation_steps"])
    use_scaler = bool(config["use_scaler"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    num_epochs = int(config["num_epochs"])

    # Get dataloader, model and trainer
    train_loader = get_dataloader(
        dataset_size=SKIPGRAM_TRAIN_DATASET_SIZE,
        num_negative_samples=num_negative_samples,
        batch_size=batch_size,
    )
    val_loader = get_dataloader(
        dataset_size=SKIPGRAM_VAL_DATASET_SIZE,
        num_negative_samples=num_negative_samples,
        batch_size=batch_size,
    )

    block_id_dict = get_block_id_dictionary()
    vocabulary_size = len(block_id_dict)
    model = _get_model(
        vocabulary_size=vocabulary_size,
        embedding_dimension=embedding_dimension,
    )

    trainer = _get_trainer(
        model=model,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        use_scaler=use_scaler,
    )

    optimizer = get_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    # Save the model and embeddings
    model.save_input_embeddings(SKIPGRAM_EMBEDDINGS_PATH)
    model.save_model(SKIPGRAM_MODEL_PATH)
