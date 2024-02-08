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

from src.utils.log import warn
from src.data.region_from_mca import get_region
from src.utils.block_dictionary import get_block_id_dictionary
from src.config import (
    CLUSTER_SIZE,
    CLUSTER_STRIDE,
    CLUSTER_DATASET_PATH,
    REGION_DATASET_PATH,
)


def _parse_arguments() -> Tuple[int, int, bool, int]:
    """
    Parse command line arguments.

    Returns:
        tuple: Tuple containing:
            cluster_size (int): Size of the clusters in blocks. Defaults to CLUSTER_SIZE.
            cluster_stride (int): Stride of the clusters in blocks. Defaults to CLUSTER_STRIDE.
            parallelize (bool): Whether to parallelize the processing of region files. Defaults to False.
            max_concurrent_processes (int): Maximum number of concurrent processes. Defaults to the number of CPU cores.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster_size",
        type=int,
        default=CLUSTER_SIZE,
        help="Size of the clusters in blocks.",
    )
    parser.add_argument(
        "--cluster_stride",
        type=int,
        default=CLUSTER_STRIDE,
        help="Stride of the clusters in blocks.",
    )
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

    cluster_size = args.cluster_size
    cluster_stride = args.cluster_stride
    parallelize = args.parallelize
    max_concurrent_processes = args.max_concurrent_processes

    if not parallelize:
        warn(
            "Processing region files without parallelization may take a long time. Consider using the --parallelize flag."
        )

    return cluster_size, cluster_stride, parallelize, max_concurrent_processes


def _get_region_names_and_paths() -> List[Tuple[str, str, str]]:
    """
    Get the paths of all region files in the dataset.

    Returns:
        List[Tuple[str, str, str]]: List of tuples containing the region folder name, region folder path, and region file name.
    """
    region_names_and_paths = []
    # Iterate through each region folder in the dataset
    for region_folder in os.listdir(REGION_DATASET_PATH):
        region_folder_path = os.path.join(REGION_DATASET_PATH, region_folder)
        if not os.path.isdir(region_folder_path):
            continue

        # Iterate through each region file in the folder
        for region_file in os.listdir(region_folder_path):
            # Add region folder name, region folder path, and region file name to region names and paths
            region_names_and_paths.append(
                (region_folder, region_folder_path, region_file)
            )

    return region_names_and_paths


def _process_region_file(
    region_folder: str,
    region_folder_path: str,
    region_file: str,
    cluster_size: int,
    cluster_stride: int,
    block_id_dict: Dict[str, int],
    parallelize: bool,
) -> None:
    """
    Process a region file.

    Args:
        region_folder (str): Name of the region folder.
        region_folder_path (str): Path to the region folder.
        region_file (str): Name of the region file.
        cluster_size (int): Size of the clusters in blocks.
        cluster_stride (int): Stride of the clusters in blocks.
        block_id_dict (Dict[str, int]): Dictionary of block states and their corresponding index.
        parallelize (bool): Whether to parallelize the processing of region files.
    """
    region_file_path = os.path.join(region_folder_path, region_file)

    # Get region
    region = get_region(
        region_file_path,
        block_id_dict=block_id_dict,
        parallelize_chunks=not parallelize,  # If already parallelized at the region level, don't parallelize at the chunk level
        show_bar=False,
    )

    # Get clusters
    relevant_clusters = region.get_clusters(
        block_id_dict=block_id_dict,
        cluster_size=cluster_size,
        cluster_stride=cluster_stride,
        only_relevant=True,
    )
    relevant_cluster_data_list = [cluster.get_data() for cluster in relevant_clusters]

    # Save clusters
    for i, relevant_cluster_data in enumerate(relevant_cluster_data_list):
        # Define the path to save the cluster data
        cluster_file_path = os.path.join(
            CLUSTER_DATASET_PATH,
            region_folder,
            f'{region_file.replace(".mca", "")}_{i}.pkl.gz',
        )

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(cluster_file_path), exist_ok=True)

        # Save the cluster data
        with gzip.open(cluster_file_path, "wb") as f:
            pickle.dump(relevant_cluster_data, f)


def _process_region_file_imap(args) -> None:
    """
    Process a region file using the imap_unordered method. Used for parallelization.

    Args:
        args (tuple): Tuple containing:
            region_folder (str): Name of the region folder.
            region_folder_path (str): Path to the region folder.
            region_file (str): Name of the region file.
            cluster_size (int): Size of the clusters in blocks.
            cluster_stride (int): Stride of the clusters in blocks.
            block_id_dict (dict): Dictionary of block states and their corresponding index.
            parallelize (bool): Whether to parallelize the processing of region files.
    """
    _process_region_file(*args)


if __name__ == "__main__":
    # Parse arguments
    (
        cluster_size,
        cluster_stride,
        parallelize,
        max_concurrent_processes,
    ) = _parse_arguments()

    # Get block id dictionary
    block_id_dict = get_block_id_dictionary()

    # Process region files
    region_names_and_paths = _get_region_names_and_paths()
    bar = tqdm(total=len(region_names_and_paths), desc="🔄 Processing region files")
    if parallelize:
        p = Pool(processes=max_concurrent_processes)

        # Get list of arguments for each process
        args_list = [
            (
                region_folder,
                region_folder_path,
                region_file,
                cluster_size,
                cluster_stride,
                block_id_dict,
                parallelize,
            )
            for region_folder, region_folder_path, region_file in region_names_and_paths
        ]

        # Process region files
        results = p.imap_unordered(_process_region_file_imap, args_list)

        # Wait for all processes to finish
        for _ in results:
            bar.update(1)

        p.close()
    else:
        for region_folder, region_folder_path, region_file in region_names_and_paths:
            _process_region_file(
                region_folder,
                region_folder_path,
                region_file,
                cluster_size,
                cluster_stride,
                block_id_dict,
                parallelize,
            )
            bar.update(1)

    bar.close()
