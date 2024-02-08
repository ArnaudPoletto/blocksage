import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import torch
import numpy as np

# Production vs development
PRODUCTION = False
PRINT_LOGS = PRODUCTION == False

# Random
SEED = 42

# Torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Wandb
WANDB_PROJECT_NAME = "blocksage"
WANDB_DATASET_NAME = "Minecraft Generated Worlds"
SWEEP_NUM_EVALUATIONS_PER_EPOCH = 25

# Paths
DATA_PATH = str(GLOBAL_DIR / "data") + "/"
CONFIG_PATH = str(GLOBAL_DIR / "config") + "/"
DATA_MODELS_PATH = str(GLOBAL_DIR / "data" / "models") + "/"

REGION_DATASET_PATH = f"{DATA_PATH}region_dataset/"
CLUSTER_DATASET_PATH = f"{DATA_PATH}cluster_dataset/"
SKIPGRAM_DATASET_PATH = f"{DATA_PATH}skipgram_dataset/"

BLOCK_STATES_PATH = f"{DATA_PATH}blockstates/"
BLOCK_ID_DICT_PATH = f"{DATA_PATH}block_id_dict.json"
BLOCK_COLOR_DICT_PATH = f"{DATA_PATH}block_color_dict.json"

TRAINER_RESULTS_FOLDER_PATH = f"{DATA_PATH}results/"
TRAINER_RESULTS_FILE_PATH = f"{TRAINER_RESULTS_FOLDER_PATH}results.csv"

SKIPGRAM_COOCCURRENCE_MATRIX_PATH = f"{DATA_PATH}skipgram_cooccurrence_matrix.npy"
SKIPGRAM_UNIGRAM_DISTRIBUTION_PATH = f"{DATA_PATH}skipgram_unigram_distribution.npy"
SKIPGRAM_CONFIG_PATH = f"{CONFIG_PATH}skipgram_best_params.yml"
SKIPGRAM_SWEEP_CONFIG_PATH = f"{CONFIG_PATH}skipgram_sweep_params.yml"
SKIPGRAM_EMBEDDINGS_PATH = f"{DATA_PATH}skipgram_embeddings.npy"
SKIPGRAM_MODEL_PATH = f"{DATA_MODELS_PATH}skipgram.pt"

# MCA file
SECTION_SIZE = 16
MIN_Y = -64 # Since 1.17
MAX_Y = 320 # Since 1.17
CHUNK_Y_SIZE = MAX_Y - MIN_Y
CHUNK_XZ_SIZE = 16
CLUSTER_SIZE = 3  # Must greater than zero, odd, and smaller or equal than N_SECTIONS_PER_CLUSTER_PER_DIM
CLUSTER_STRIDE = 3  # Set equal to CLUSTER_SIZE for non-overlapping in x z directions clusters is best
N_SECTIONS_PER_CLUSTER_PER_DIM = CHUNK_Y_SIZE // SECTION_SIZE
N_CHUNKS_PER_REGION_PER_DIM = 32

# Skipgram
SKIPGRAM_NAME = "skipgram"
SKIPGRAM_WINDOW_SIZE = 1
SKIPGRAM_TRAIN_DATASET_SIZE = 1_000_000
SKIPGRAM_VAL_DATASET_SIZE = 10_000
SKIPGRAM_NUM_WORKERS = 0

# Color
BLACK_COLOR = [0, 0, 0]

# Block
MASKED_BLOCK_ID = np.uint16(-1)

AIR_NAME = "air"
CAVE_AIR_NAME = "cave_air"
VOID_AIR_NAME = "void_air"

WATER_NAME = "water"
LAVA_NAME = "lava"

STONE_NAME = "stone"
DEEPSLATE_NAME = "deepslate"
ANDESITE_NAME = "andesite"
DIORITE_NAME = "diorite"
GRANITE_NAME = "granite"
TUFF_NAME = "tuff"
BEDROCK_NAME = "bedrock"

GRASS_BLOCK_NAME = "grass_block"
GRAVEL_NAME = "gravel"

COAL_ORE_NAME = "coal_ore"
COPPER_ORE_NAME = "copper_ore"
IRON_ORE_NAME = "iron_ore"
GOLD_ORE_NAME = "gold_ore"
DIAMOND_ORE_NAME = "diamond_ore"
EMERALD_ORE_NAME = "emerald_ore"
LAPIS_ORE_NAME = "lapis_ore"
REDSTONE_ORE_NAME = "redstone_ore"
DEEPSLATE_COAL_ORE_NAME = "deepslate_coal_ore"
DEEPSLATE_COPPER_ORE_NAME = "deepslate_copper_ore"
DEEPSLATE_IRON_ORE_NAME = "deepslate_iron_ore"
DEEPSLATE_GOLD_ORE_NAME = "deepslate_gold_ore"
DEEPSLATE_DIAMOND_ORE_NAME = "deepslate_diamond_ore"
DEEPSLATE_EMERALD_ORE_NAME = "deepslate_emerald_ore"
DEEPSLATE_LAPIS_ORE_NAME = "deepslate_lapis_ore"
DEEPSLATE_REDSTONE_ORE_NAME = "deepslate_redstone_ore"

NATURAL_UNDERGROUND_BLOCK_NAMES = [
    CAVE_AIR_NAME,
    WATER_NAME,
    LAVA_NAME,
    STONE_NAME,
    DEEPSLATE_NAME,
    ANDESITE_NAME,
    DIORITE_NAME,
    GRANITE_NAME,
    BEDROCK_NAME,
    GRASS_BLOCK_NAME,
    GRAVEL_NAME,
    COAL_ORE_NAME,
    COPPER_ORE_NAME,
    IRON_ORE_NAME,
    GOLD_ORE_NAME,
    DIAMOND_ORE_NAME,
    EMERALD_ORE_NAME,
    LAPIS_ORE_NAME,
    REDSTONE_ORE_NAME,
    DEEPSLATE_COAL_ORE_NAME,
    DEEPSLATE_COPPER_ORE_NAME,
    DEEPSLATE_IRON_ORE_NAME,
    DEEPSLATE_GOLD_ORE_NAME,
    DEEPSLATE_DIAMOND_ORE_NAME,
    DEEPSLATE_EMERALD_ORE_NAME,
    DEEPSLATE_LAPIS_ORE_NAME,
    DEEPSLATE_REDSTONE_ORE_NAME,
]
