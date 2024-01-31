from src.config import PRINT_LOGS


def log(*args: list, **kwargs: dict) -> None:
    """
    Print a message to the console if PRINT_LOGS is True.

    Args:
        *args (list): List of arguments to pass to print().
        **kwargs (dict): Dictionary of keyword arguments to pass to print().
    """
    if PRINT_LOGS:
        print(*args, **kwargs)


def warn(*args: list, **kwargs: dict) -> None:
    """
    Print a warning message to the console if PRINT_LOGS is True.

    Args:
        *args (list): List of arguments to pass to print().
        **kwargs (dict): Dictionary of keyword arguments to pass to print().
    """
    if PRINT_LOGS:
        print("⚠️ ", *args, **kwargs)
