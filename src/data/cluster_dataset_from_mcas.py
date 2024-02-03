import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

from src.utils.log import warn
from src.data.region_from_mca import get_region
from src.utils.block_dictionary import get_block_id_dictionary
from src.config import (
    CLUSTER_SIZE, 
    CLUSTER_STRIDE,
    CLUSTER_DATASET_PATH,
    REGION_DATASET_PATH
)


def parse_arguments():
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


def process_region_file(
    region_folder: str,
    region_folder_path: str,
    region_file: str,
    cluster_size: int,
    cluster_stride: int,
    block_id_dict: dict,
    parallelize: bool,
):
    """
    Process a region file.

    Args:
        region_folder (str): Name of the region folder.
        region_folder_path (str): Path to the region folder.
        region_file (str): Name of the region file.
        cluster_size (int): Size of the clusters in blocks.
        cluster_stride (int): Stride of the clusters in blocks.
        block_id_dict (dict): Dictionary of block states and their corresponding index.
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
        stride=cluster_stride,
        only_relevant=True,
    )
    relevant_cluster_data_list = [cluster.get_data() for cluster in relevant_clusters]

    # Save clusters
    for i, relevant_cluster_data in enumerate(relevant_cluster_data_list):
        # Define the path to save the cluster data
        cluster_file_path = os.path.join(
            CLUSTER_DATASET_PATH,
            region_folder,
            f'{region_file.replace(".mca", "")}_{i}.npy',
        )

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(cluster_file_path), exist_ok=True)

        # Save the cluster data
        np.save(cluster_file_path, relevant_cluster_data)


if __name__ == "__main__":
    # Parse arguments
    (
        cluster_size,
        cluster_stride,
        parallelize,
        max_concurrent_processes,
    ) = parse_arguments()

    # Get block id dictionary
    block_id_dict = get_block_id_dictionary()

    with ProcessPoolExecutor(max_workers=max_concurrent_processes) as executor:
        futures = []
        region_folders = os.listdir(REGION_DATASET_PATH)
        total_files = sum(
            len(os.listdir(os.path.join(REGION_DATASET_PATH, region_folder)))
            for region_folder in os.listdir(REGION_DATASET_PATH)
        )
        bar = tqdm(total=total_files, desc="🔄 Processing region file")
        for region_folder in region_folders:
            region_folder_path = os.path.join(REGION_DATASET_PATH, region_folder)
            if not os.path.isdir(region_folder_path):
                continue

            for region_file in os.listdir(region_folder_path):
                if parallelize:
                    futures.append(
                        executor.submit(
                            process_region_file,
                            region_folder,
                            region_folder_path,
                            region_file,
                            cluster_size,
                            cluster_stride,
                            block_id_dict,
                            parallelize,
                        )
                    )
                else:
                    process_region_file(
                        region_folder,
                        region_folder_path,
                        region_file,
                        cluster_size,
                        cluster_stride,
                        block_id_dict,
                        parallelize,
                    )
                    bar.update(1)

        for future in as_completed(futures):
            future.result()
            bar.update(1)
