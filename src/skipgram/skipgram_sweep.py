import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import wandb
from typing import Dict

from src.utils.parser import get_config
from src.sweepers.skipgram_sweeper import SkipGramSweeper
from src.models.skipgram import SkipGram
from src.datasets.skipgram_dataset import get_dataloader
from src.utils.block_dictionary import get_block_id_dictionary
from src.config import (
    SKIPGRAM_TRAIN_DATASET_SIZE,
    SKIPGRAM_VAL_DATASET_SIZE,
    SKIPGRAM_SWEEP_CONFIG_PATH,
    WANDB_PROJECT_NAME,
)


def _sweep(config: Dict = None) -> None:
    """
    Sweep the model.

    Args:
        config (Dict, optional): The config. Defaults to None.
    """
    with wandb.init(config=config):
        config = wandb.config
        # Get dataloader parameters from config
        num_negative_samples = config.num_negative_samples
        batch_size = config.batch_size
        noise_power = config.noise_power
        minimum_noise_distribution = config.minimum_noise_distribution
        subsampling_threshold = config.subsampling_threshold
        minimum_subsampling_distribution = config.minimum_subsampling_distribution

        # Get dataloaders
        train_dataloader = get_dataloader(
            dataset_size=SKIPGRAM_TRAIN_DATASET_SIZE,
            num_negative_samples=num_negative_samples,
            batch_size=batch_size,
            noise_power=noise_power,
            minimum_noise_distribution=minimum_noise_distribution,
            subsampling_threshold=subsampling_threshold,
            minimum_subsampling_distribution=minimum_subsampling_distribution,
        )
        val_dataloader = get_dataloader(
            dataset_size=SKIPGRAM_VAL_DATASET_SIZE,
            num_negative_samples=num_negative_samples,
            batch_size=batch_size,
            noise_power=noise_power,
            minimum_noise_distribution=minimum_noise_distribution,
            subsampling_threshold=subsampling_threshold,
            minimum_subsampling_distribution=minimum_subsampling_distribution,
        )

        # Get block id dictionary and vocabulary size
        block_id_dict = get_block_id_dictionary()
        vocabulary_size = len(block_id_dict)

        # Sweep the model
        sweeper = SkipGramSweeper(
            model_class=SkipGram,
            config=config,
            vocabulary_size=vocabulary_size,
        )
        sweeper.train(train_loader=train_dataloader, val_loader=val_dataloader)


if __name__ == "__main__":
    config = get_config(SKIPGRAM_SWEEP_CONFIG_PATH)
    sweep_id = wandb.sweep(config, project=WANDB_PROJECT_NAME)
    wandb.agent(sweep_id, function=_sweep)
