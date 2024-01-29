import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import json
import numpy as np

from src.utils.log import log

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
BLOCK_STATES_PATH = DATA_PATH + "blockstates/"
BLOCK_DICT_PATH = DATA_PATH + "block_id_dict.json"
BLOCK_COLOR_DICT_PATH = DATA_PATH + "block_color_dict.json"


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

    with open(BLOCK_DICT_PATH, "w") as f:
        json.dump(block_dict, f)


def get_block_id_dictionary() -> dict:
    """
    Get a dictionary of block states and their corresponding index.

    Returns:
        dict: Dictionary of block states and their corresponding index, e.g. {'minecraft:air': 0, 'minecraft:stone': 1, ...}
    """
    with open(BLOCK_DICT_PATH) as f:
        block_dict = json.load(f)

    log(
        f"✅ Loaded {len(block_dict)} block states from {Path(BLOCK_DICT_PATH).resolve()}."
    )

    return block_dict


def get_block_color_dictionary() -> dict:
    """
    Get a dictionary of block states and their corresponding color.

    Returns:
        dict: Dictionary of block states and their corresponding color, e.g. {'minecraft:air': [0, 0, 0], 'minecraft:stone': [0.5, 0.5, 0.5], ...}
    """
    with open(BLOCK_COLOR_DICT_PATH) as f:
        block_color_dict = json.load(f)

    log(
        f"✅ Loaded {len(block_color_dict)} block colors from {Path(BLOCK_COLOR_DICT_PATH).resolve()}."
    )

    return block_color_dict
