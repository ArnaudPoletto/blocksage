import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from src.config import DATA_PATH, DATASET_SUBSET_FRACTION, DEVICE
from src.datasets.cluster_dataset import get_dataloaders
from src.utils.block_dictionary import get_block_id_dictionary


MAX_NUM_SAMPLES = 100
CE_CLASS_WEIGHTS_PATH = f"{DATA_PATH}ce_class_weights.npy"

def _compute_and_get_ce_class_weights(
        block_id_dict: dict, 
        block_class_dict: dict = None,
        max_num_samples: int = MAX_NUM_SAMPLES,
        ) -> None:
    # Get dataloader and number of block classes
    train_dataloader, _, _ = get_dataloaders(
        block_id_dict = block_id_dict,
        block_class_dict = block_class_dict,
        subset_fraction = DATASET_SUBSET_FRACTION
        )
    
    # The number of block classes is the number of unique block ids if no block class dictionary is provided
    # Otherwise, it is the number of unique block classes
    # The number of total classes is the number of block classes + 1 for the masked block
    num_block_classes = len(block_id_dict) if block_class_dict is None else np.unique(list(block_class_dict.values())).shape[0]
    num_total_classes = num_block_classes + 1 # +1 for the masked block

    ce_class_weights = torch.zeros(num_total_classes)
    for i, batch in tqdm(enumerate(train_dataloader), total=max_num_samples, desc="🔄 Computing class weights"):
        _, cluster_gt = batch
        ce_class_weights += torch.sum(cluster_gt, dim=(2, 3, 4))[0]

        if i == max_num_samples:
            break
    
    ce_class_weights = ce_class_weights / torch.sum(ce_class_weights)
    ce_class_weights = torch.where(ce_class_weights == 0, 0, 1 / ce_class_weights)
    ce_class_weights_sum = torch.sum(ce_class_weights)
    ce_class_weights = torch.where(ce_class_weights == 0, 1, ce_class_weights / ce_class_weights_sum)

    # Save class weights
    np.save(CE_CLASS_WEIGHTS_PATH, ce_class_weights.numpy())

    return ce_class_weights


def _get_ce_class_weights(
        block_id_dict: dict,
        block_class_dict: dict = None,
    ) -> torch.Tensor:
    if Path(CE_CLASS_WEIGHTS_PATH).exists():
        ce_class_weights = torch.from_numpy(np.load(CE_CLASS_WEIGHTS_PATH))
    else:
        ce_class_weights = _compute_and_get_ce_class_weights(
            block_id_dict = block_id_dict, 
            block_class_dict = block_class_dict
        )

    print(np.unique(ce_class_weights.numpy(), return_counts=True))

    return ce_class_weights.float().to(DEVICE)


def get_criterion(
        criterion_name: str, 
        block_id_dict: dict = None,
        block_class_dict: dict = None,
    ) -> nn.Module:
    # Get block id dictionary if not provided
    if block_id_dict is None:
        block_id_dict = get_block_id_dictionary()

    # Get criterion
    if criterion_name == 'ce':
        ce_class_weights = _get_ce_class_weights(
            block_id_dict = block_id_dict,
            block_class_dict = block_class_dict
        )

        return nn.CrossEntropyLoss(weight=ce_class_weights)
    else:
        raise NotImplementedError(f"❌ Criterion {criterion_name} not regcognized.")
    
def get_optimizer(model, learning_rate, weight_decay):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)