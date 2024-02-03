import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from multiprocessing import Pool
from src.utils.log import log, warn

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


def _parse_arguments() -> Tuple[bool, int]:
    """
    Parse command line arguments.
    
    Returns:
        tuple: Tuple containing:
            parallelize (bool): Whether to parallelize the processing of region files. Defaults to False.
            max_concurrent_processes (int): Maximum number of concurrent processes. Defaults to the number of CPU cores.
    """
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

    if not parallelize:
        warn(
            "Processing cluster files without parallelization may take a long time. Consider using the --parallelize flag."
        )

    return parallelize, max_concurrent_processes


def _get_cluster_file_paths() -> List[str]:
    """
    Get the paths of all cluster files in the dataset.

    Returns:
        List[str]: List of paths of all cluster files in the dataset.
    """
    cluster_file_paths = []
    # Iterate through each cluster folder in the dataset
    for cluster_folder in os.listdir(CLUSTER_DATASET_PATH):
        cluster_folder_path = os.path.join(CLUSTER_DATASET_PATH, cluster_folder)
        if not os.path.isdir(cluster_folder_path):
            continue

        # Iterate through each cluster file in the folder
        for cluster_file in os.listdir(cluster_folder_path):
            cluster_file_path = os.path.join(cluster_folder_path, cluster_file)

            # Add cluster file path to cluster paths
            cluster_file_paths.append(cluster_file_path)

    return cluster_file_paths


def _process_cluster_file(
    cluster_file_path: str,
    vocabulary_size: int,
    section_size: int,
    cluster_size: int,
    skipgram_window_size: int,
) -> np.ndarray:
    """
    Process a cluster file to obtain a sub cooccurrence matrix.

    Args:
        cluster_file_path (str): Path to the cluster file.
        vocabulary_size (int): Size of the vocabulary.
        section_size (int): Size of the section.
        cluster_size (int): Size of the cluster.
        skipgram_window_size (int): Size of the skipgram window.

    Returns:
        np.ndarray: Sub cooccurrence matrix.
    """
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

def _process_cluster_file_imap(args) -> np.ndarray:
    """
    Process a cluster file to obtain a sub cooccurrence matrix.

    Args:
        args (tuple): Tuple containing:
            cluster_file_path (str): Path to the cluster file.
            vocabulary_size (int): Size of the vocabulary.
            section_size (int): Size of the section.
            cluster_size (int): Size of the cluster.
            skipgram_window_size (int): Size of the skipgram window.

    Returns:
        np.ndarray: Sub cooccurrence matrix.
    """
    return _process_cluster_file(*args)


if __name__ == "__main__":
    # Parse arguments
    parallelize, max_concurrent_processes = _parse_arguments()

    # Get block id dictionary and vocabulary size
    block_id_dict = get_block_id_dictionary()
    vocabulary_size = len(block_id_dict)

    # Process cluster files
    cooccurrence_matrix = np.zeros(
        (vocabulary_size, vocabulary_size), dtype=np.float32
    )
    cluster_file_paths = _get_cluster_file_paths()
    if parallelize:
        p = Pool(processes=max_concurrent_processes)

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
        results = p.imap_unordered(_process_cluster_file_imap, args_list)

        # Aggregate results
        for result in tqdm(results, total=len(args_list), desc="🔄 Processing cluster files"):
            cooccurrence_matrix += result 

        p.close()
    else:
        for cluster_file_path in tqdm(cluster_file_paths, desc="🔄 Processing cluster files"):
            cooccurrence_matrix += _process_cluster_file(
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
