import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
from typing import List

from src.config import CLUSTER_DATASET_PATH

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

if __name__ == "__main__":
    pass