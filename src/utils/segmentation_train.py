import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from src.config import DATA_PATH, DEVICE
from src.datasets.cluster_dataset import get_dataloaders
from src.utils.block_dictionary import get_block_id_dictionary


CE_CLASS_WEIGHTS_PATH = f"{DATA_PATH}ce_class_weights.npy"

def _compute_and_get_ce_class_weights(block_id_dict: dict, eps: float = 1e-5) -> None:
    # Get dataloader and number of block classes
    train_dataloader, _, _ = get_dataloaders(block_id_dict=block_id_dict)
    num_block_classes = len(block_id_dict)

    ce_class_weights = torch.zeros(num_block_classes + 1)
    for batch in tqdm(train_dataloader, desc="🔄 Computing class weights"):
        _, cluster_gt = batch
        ce_class_weights += torch.sum(cluster_gt, dim=(2, 3, 4))[0]
        
    ce_class_weights = (ce_class_weights + eps) / (torch.sum(ce_class_weights) + eps)
    ce_class_weights = eps / (ce_class_weights + eps)

    # Save class weights
    np.save(CE_CLASS_WEIGHTS_PATH, ce_class_weights.numpy())

    return ce_class_weights


def _get_ce_class_weights(block_id_dict: dict) -> torch.Tensor:
    if Path(CE_CLASS_WEIGHTS_PATH).exists():
        ce_class_weights = torch.from_numpy(np.load(CE_CLASS_WEIGHTS_PATH))
    else:
        ce_class_weights = _compute_and_get_ce_class_weights(block_id_dict=block_id_dict)

    return ce_class_weights.float().to(DEVICE)


def get_criterion(criterion_name: str, block_id_dict: dict = None) -> nn.Module:
    # Get block id dictionary if not provided
    if block_id_dict is None:
        block_id_dict = get_block_id_dictionary()

    # Get criterion
    if criterion_name == 'ce':
        ce_class_weights = _get_ce_class_weights(block_id_dict)
        return nn.CrossEntropyLoss(weight=ce_class_weights)
    
def get_optimizer(model, learning_rate, weight_decay):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)