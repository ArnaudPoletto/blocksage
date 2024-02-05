import yaml
from typing import Dict
from pathlib import Path


def get_config(config_path: str) -> Dict:
    """
    Get the config.

    Args:
        config_path (str): The path to the config file.

    Returns:
        Dict: The config as a dictionary.
    """
    config = yaml.safe_load(Path(config_path).read_text())

    return config
