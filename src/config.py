import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".."
sys.path.append(str(GLOBAL_DIR))

import torch

# Production vs development
PRODUCTION = False

# Paths
DATA_PATH = str(GLOBAL_DIR / "data") + "/"
REGION_DATASET_PATH = f"{DATA_PATH}region_dataset/"
CLUSTER_DATASET_PATH = f"{DATA_PATH}cluster_dataset/"
SKIPGRAM_DATASET_PATH = f"{DATA_PATH}skipgram_dataset/"
SKIPGRAM_COOCCURRENCE_MATRIX_PATH = f"{DATA_PATH}skipgram_cooccurrence_matrix.npy"

# Random
SEED = 42

# Torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Log
PRINT_LOGS = PRODUCTION == False

# Wandb
WANDB_PROJECT_NAME = "blocksage"

# MCA file
SECTION_SIZE = 16

MIN_Y = -64
MAX_Y = 320
CHUNK_Y_SIZE = MAX_Y - MIN_Y
CHUNK_XZ_SIZE = 16

CLUSTER_SIZE = 3
CLUSTER_STRIDE = 1
MAX_N_SECTIONS_PER_CLUSTER_PER_DIM = CHUNK_Y_SIZE // SECTION_SIZE

N_CHUNKS_PER_REGION_PER_DIM = 32

# Dataset and dataloader
DATASET_SUBSET_FRACTION = 1.0 if PRODUCTION else 0.01
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05
BATCH_SIZE = 2
NUM_WORKERS = 0

# Model
SKIPGRAM_WINDOW_SIZE = 2
ENCODER_CONV_CHANNELS = [512, 128, 32]
DECODER_CONV_CHANNELS = [32, 128, 512]

# Color
BLACK_COLOR = [0, 0, 0]

# Block
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
