import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import gzip
import pickle
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
    SKIPGRAM_UNIGRAM_DISTRIBUTION_PATH,
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
) -> np.ndarray:
    """
    Process a cluster file to obtain sub block counts.

    Args:
        cluster_file_path (str): Path to the cluster file.
        vocabulary_size (int): Size of the vocabulary.

    Returns:
        np.ndarray: Sub block counts.
    """
    # Get cluster
    with gzip.open(cluster_file_path, "rb") as f:
        cluster_data = pickle.load(f)

    # Get sub block counts
    sub_block_counts = np.zeros((vocabulary_size,), dtype=np.uint32)
    unique_block_ids, unique_block_counts = np.unique(cluster_data, return_counts=True)
    sub_block_counts[unique_block_ids] = unique_block_counts

    return sub_block_counts


def _process_cluster_file_imap(args) -> np.ndarray:
    """
    Process a cluster file to obtain sub block counts. Used for parallelization.

    Args:
        args (tuple): Tuple containing:
            cluster_file_path (str): Path to the cluster file.
            vocabulary_size (int): Size of the vocabulary.
            section_size (int): Size of the section.
            cluster_size (int): Size of the cluster.
            skipgram_window_size (int): Size of the skipgram window.

    Returns:
        np.ndarray: Sub block counts.
    """
    return _process_cluster_file(*args)


if __name__ == "__main__":
    # Parse arguments
    parallelize, max_concurrent_processes = _parse_arguments()

    # Get block id dictionary and vocabulary size
    block_id_dict = get_block_id_dictionary()
    vocabulary_size = len(block_id_dict)

    # Process cluster files
    block_counts = np.zeros((vocabulary_size,), dtype=np.float32)
    cluster_file_paths = _get_cluster_file_paths()
    if parallelize:
        p = Pool(processes=max_concurrent_processes)

        # Get list of arguments for each process
        args_list = [
            (
                cluster_file_path,
                vocabulary_size,
            )
            for cluster_file_path in cluster_file_paths
        ]

        # Process cluster files
        results = p.imap_unordered(_process_cluster_file_imap, args_list)

        # Aggregate results
        for result in tqdm(
            results, total=len(args_list), desc="🔄 Processing cluster files"
        ):
            block_counts += result

        p.close()
    else:
        for cluster_file_path in tqdm(
            cluster_file_paths, desc="🔄 Processing cluster files"
        ):
            block_counts += _process_cluster_file(
                cluster_file_path,
                vocabulary_size,
            )

    # Get and save the unigram distribution
    unigram_distribution = block_counts / block_counts.sum()
    np.save(SKIPGRAM_UNIGRAM_DISTRIBUTION_PATH, unigram_distribution)
    log(
        f"💾 Saved unigram distribution to {Path(SKIPGRAM_UNIGRAM_DISTRIBUTION_PATH).resolve()}"
    )
