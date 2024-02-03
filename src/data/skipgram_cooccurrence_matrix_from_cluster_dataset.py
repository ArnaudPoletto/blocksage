import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from src.utils.log import log, warn
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from src.zones.cluster import Cluster
from src.utils.block_dictionary import get_block_id_dictionary
from src.config import (
    CLUSTER_DATASET_PATH,
    CLUSTER_SIZE,
    SECTION_SIZE,
    SKIPGRAM_WINDOW_SIZE,
    SKIPGRAM_COOCCURRENCE_MATRIX_PATH,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help="Whether to parallelize the processing of region files.",
    )
    parser.add_argument(
        "--max_concurrent_processes",
        type=int,
        default=os.cpu_count(),
        help="Maximum number of concurrent processes.",
    )
    args = parser.parse_args()

    parallelize = args.parallelize
    max_concurrent_processes = args.max_concurrent_processes

    return parallelize, max_concurrent_processes


def get_cluster_file_paths() -> List[str]:
    cluster_file_paths = []
    # Iterate through each cluster folder in the dataset
    for cluster_folder in os.listdir(CLUSTER_DATASET_PATH):
        cluster_folder_path = os.path.join(CLUSTER_DATASET_PATH, cluster_folder)
        if not os.path.isdir(cluster_folder_path):
            continue

        # Iterate through each region file in the folder
        for cluster_file in os.listdir(cluster_folder_path):
            cluster_file_path = os.path.join(cluster_folder_path, cluster_file)

            # Add cluster file path to cluster paths
            cluster_file_paths.append(cluster_file_path)

    return cluster_file_paths


def process_cluster_file(
    cluster_file_path: str,
    vocabulary_size: int,
    section_size: int,
    cluster_size: int,
    skipgram_window_size: int,
):
    # Get cluster
    cluster_data = np.load(cluster_file_path)
    cluster = Cluster(cluster_data)
    cluster_data = cluster.get_data_by_cluster()

    # Get the center section as target blocks
    center_section_idx = (section_size // 2) * cluster_size
    target_blocks = cluster_data[
        center_section_idx : center_section_idx + section_size,
        center_section_idx : center_section_idx + section_size,
        center_section_idx : center_section_idx + section_size,
    ]

    # Get the context blocks
    # First, get the list of shifts to apply to the center section
    shifts = np.array(
        np.meshgrid(
            *[np.arange(-skipgram_window_size, skipgram_window_size + 1)] * 3,
            indexing="ij",
        )
    ).T.reshape(-1, 3)
    shifts = np.delete(
        shifts, shifts.shape[0] // 2, axis=0
    )  # Delete the shift [0, 0, 0]

    # Then, get the context blocks
    context_blocks = np.zeros(
        (len(shifts), section_size, section_size, section_size), dtype=np.uint16
    )
    for i, shift in enumerate(shifts):
        context_blocks[i] = np.roll(cluster_data, shift, axis=(0, 1, 2))[
            center_section_idx : center_section_idx + section_size,
            center_section_idx : center_section_idx + section_size,
            center_section_idx : center_section_idx + section_size,
        ]

    # Then, get the number of cooccurrences between each target and context block
    target_blocks = target_blocks.flatten()
    context_blocks = context_blocks.reshape(len(shifts), -1)
    target_indices = np.repeat(target_blocks, len(shifts))
    context_indices = context_blocks.ravel()
    block_id_pairs = np.stack((target_indices, context_indices), axis=1)
    sub_cooccurrence_pair_idx, sub_cooccurrence_pair_freq = np.unique(
        block_id_pairs, axis=0, return_counts=True
    )
    sub_cooccurrence_pair_freq = sub_cooccurrence_pair_freq.astype(np.uint32)

    # Finally, get the sub cooccurrence matrix
    sub_cooccurrence_matrix = np.zeros(
        (vocabulary_size, vocabulary_size), dtype=np.uint32
    )
    sub_cooccurrence_matrix[
        sub_cooccurrence_pair_idx[:, 0], sub_cooccurrence_pair_idx[:, 1]
    ] += sub_cooccurrence_pair_freq

    return sub_cooccurrence_matrix

def process_cluster_file_imap(args):
    return process_cluster_file(*args)


if __name__ == "__main__":
    # Parse arguments
    parallelize, max_concurrent_processes = parse_arguments()
    if not parallelize:
        warn(
            "Processing cluster files without parallelization may take a long time. Consider using the --parallelize flag."
        )

    # Get block id dictionary and vocabulary size
    block_id_dict = get_block_id_dictionary()
    vocabulary_size = len(block_id_dict)

    with Pool(processes=max_concurrent_processes) as p:
        cooccurrence_matrix = np.zeros(
            (vocabulary_size, vocabulary_size), dtype=np.float32
        )
        cluster_file_paths = get_cluster_file_paths()
        if parallelize:
            # Get list of arguments for each process
            args_list = [
                (
                    cluster_file_path,
                    vocabulary_size,
                    SECTION_SIZE,
                    CLUSTER_SIZE,
                    SKIPGRAM_WINDOW_SIZE,
                )
                for cluster_file_path in cluster_file_paths
            ]

            # Process cluster files
            results = p.imap_unordered(process_cluster_file_imap, args_list)

            # Aggregate results
            for result in tqdm(results, total=len(args_list), desc="🔄 Processing cluster files"):
                cooccurrence_matrix += result 
        else:
            for cluster_file_path in tqdm(cluster_file_paths, desc="🔄 Processing cluster files"):
                cooccurrence_matrix += process_cluster_file(
                    cluster_file_path,
                    vocabulary_size,
                    SECTION_SIZE,
                    CLUSTER_SIZE,
                    SKIPGRAM_WINDOW_SIZE,
                )

        # Normalize the cooccurrence matrix
        coocurrence_matrix_sum = np.sum(cooccurrence_matrix, axis=1, keepdims=True) # Get the number of cooccurrences for each target block
        zero_sum_indices = np.where(coocurrence_matrix_sum == 0)[0]
        cooccurrence_matrix[zero_sum_indices, :] = 1 / vocabulary_size # Target blocks with no cooccurrences are assigned a uniform probability
        coocurrence_matrix_sum[coocurrence_matrix_sum == 0] = 1  # The sum of such probabilities is 1
        cooccurrence_matrix /= coocurrence_matrix_sum

        # Save the cooccurrence matrix in a new file
        np.save(SKIPGRAM_COOCCURRENCE_MATRIX_PATH, cooccurrence_matrix)
        log(f"💾 Saved cooccurrence matrix to {SKIPGRAM_COOCCURRENCE_MATRIX_PATH}.")
