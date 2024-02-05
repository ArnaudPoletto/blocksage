import sys
from typing import List, Dict
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import json

from src.utils.log import log
from src.config import (
    BLOCK_STATES_PATH,
    BLOCK_ID_DICT_PATH,
    BLOCK_COLOR_DICT_PATH,
)


def save_block_id_dictionary() -> None:
    """
    Get a dictionary of block states and their corresponding index.
    """
    # Get list of json files in block states folder
    block_files = [f.replace(".json", "") for f in os.listdir(BLOCK_STATES_PATH)]
    block_dict = {block_files[i]: i for i in range(len(block_files))}

    log(
        f"✅ Found {len(block_dict)} block states in {Path(BLOCK_STATES_PATH).resolve()}."
    )

    with open(BLOCK_ID_DICT_PATH, "w") as f:
        json.dump(block_dict, f)


def get_block_id_dictionary() -> Dict[str, int]:
    """
    Get a dictionary of block states and their corresponding index.

    Returns:
        Dict[str, int]: Dictionary of block states and their corresponding index, e.g. {'minecraft:air': 0, 'minecraft:stone': 1, ...}.
    """
    with open(BLOCK_ID_DICT_PATH) as f:
        block_dict = json.load(f)

    log(
        f"✅ Loaded {len(block_dict)} block states from {Path(BLOCK_ID_DICT_PATH).resolve()}."
    )

    return block_dict


def get_block_color_dictionary() -> Dict[str, List[int]]:
    """
    Get a dictionary of block states and their corresponding color.

    Returns:
        Dict[str, List[int]]: Dictionary of block states and their corresponding color, e.g. {'minecraft:air': [0, 0, 0], 'minecraft:stone': [128, 128, 128], ...}.
    """
    with open(BLOCK_COLOR_DICT_PATH) as f:
        block_color_dict = json.load(f)

    log(
        f"✅ Loaded {len(block_color_dict)} block colors from {Path(BLOCK_COLOR_DICT_PATH).resolve()}."
    )

    return block_color_dict
