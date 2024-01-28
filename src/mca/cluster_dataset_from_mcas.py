import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import numpy as np
from tqdm import tqdm

from src.mca.region_from_mca import get_region
from src.utils.block_dictionary import get_block_id_dictionary

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
REGION_DATASET_PATH = f"{DATA_PATH}region_dataset/"
CLUSTER_DATASET_PATH = f"{DATA_PATH}cluster_dataset/"

CLUSTER_SIZE = 3
CLUSTER_STRIDE = 1

if __name__ == "__main__":
    # Get block id dictionary
    block_id_dict = get_block_id_dictionary()

    # Iterate through each folder containing region files
    for region_folder in tqdm(
        os.listdir(REGION_DATASET_PATH), desc="🔄 Processing region folder"
    ):
        region_folder_path = os.path.join(REGION_DATASET_PATH, region_folder)
        if os.path.isdir(region_folder_path):
            # Iterate through each region file in the folder
            for region_file in os.listdir(region_folder_path):
                region_file_path = os.path.join(region_folder_path, region_file)

                # Get region
                region = get_region(
                    region_file_path,
                    block_id_dict=block_id_dict,
                    parallelize_chunks=True,
                    show_bar=False,
                )

                # Get clusters
                relevant_clusters = region.get_clusters(
                    block_id_dict=block_id_dict,
                    cluster_size=CLUSTER_SIZE,
                    stride=CLUSTER_STRIDE,
                    only_relevant=True,
                )
                relevant_cluster_data_list = [
                    cluster.get_data_by_cluster() for cluster in relevant_clusters
                ]

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
