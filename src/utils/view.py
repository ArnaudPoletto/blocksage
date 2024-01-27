SHAPE_SIZE = 6


def view_by_section(region_blocks):
    """
    View the blocks by section, i.e. as an array of shape (region_x, region_z, section, section_x, section_y, section_z).

    Args:
        region_blocks (np.ndarray): Array of block IDs of shape (region_x, region_z, section, section_y, section_z, section_x).

    Returns:
        np.ndarray: Array of block IDs of shape (region_x, region_z, section, section_x, section_y, section_z).
    """
    if len(region_blocks.shape) != SHAPE_SIZE:
        raise ValueError(
            f"❌ region_blocks must be of shape (region_x, region_z, section, section_y, section_z, section_x), not {region_blocks.shape}."
        )

    return region_blocks.transpose((0, 1, 2, 5, 3, 4))


def view_by_chunk(region_blocks):
    """
    View the blocks by chunk, i.e. as an array of shape (region_x, region_z, chunk_x, chunk_y, chunk_z) = (region_x, region_z, section_x, section * section_y, section_z).

    Args:
        region_blocks (np.ndarray): Array of block IDs of shape (region_x, region_z, section, section_y, section_z, section_x).

    Returns:
        np.ndarray: Array of block IDs of shape (region_x, region_z, chunk_x, chunk_y, chunk_z) = (region_x, region_z, section_x, section * section_y, section_z).
    """
    if len(region_blocks.shape) != SHAPE_SIZE:
        raise ValueError(
            f"❌ region_blocks must be of shape (region_x, region_z, section, section_y, section_z, section_x), not {region_blocks.shape}."
        )

    region_x, region_z, section, section_y, section_z, section_x = region_blocks.shape
    return region_blocks \
        .reshape((region_x, region_z, section * section_y, section_z, section_x)) \
        .transpose((0, 1, 4, 2, 3))


def view_by_region(region_blocks):
    """
    View the blocks by region, i.e. as an array of shape (region_x, region_y, region_z) = (region_x * section_x, section * section_y, region_z * section_z).

    Args:
        region_blocks (np.ndarray): Array of block IDs of shape (region_x, region_z, section, section_y, section_z, section_x).

    Returns:
        np.ndarray: Array of block IDs of shape (region_x, region_y, region_z) = (region_x * section_x, section * section_y, region_z * section_z).
    """
    if len(region_blocks.shape) != SHAPE_SIZE:
        raise ValueError(
            f"❌ region_blocks must be of shape (chunk_x, chunk_z, section, section_y, section_z, section_x), not {region_blocks.shape}."
        )

    region_x, region_z, section, section_y, section_z, section_x = region_blocks.shape
    return (
        region_blocks.transpose(0, 5, 1, 4, 2, 3)
        .reshape((region_x * section_x, region_z * section_z, section * section_y))
        .transpose(0, 2, 1)
    )
