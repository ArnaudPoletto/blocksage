from typing import Tuple, Dict

from src.config import PRINT_LOGS


def log(*args: Tuple, **kwargs: Dict) -> None:
    """
    Print a message to the console if PRINT_LOGS is True.

    Args:
        *args (Tuple): List of arguments to pass to print().
        **kwargs (Dict): Dictionary of keyword arguments to pass to print().
    """
    if PRINT_LOGS:
        print(*args, **kwargs)


def warn(*args: Tuple, **kwargs: Dict) -> None:
    """
    Print a warning message to the console if PRINT_LOGS is True.

    Args:
        *args (Tuple): List of arguments to pass to print().
        **kwargs (Dict): Dictionary of keyword arguments to pass to print().
    """
    if PRINT_LOGS:
        print("⚠️ ", *args, **kwargs)
