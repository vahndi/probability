from typing import Optional


def distribute_about_center(index: int, size: int,
                            max_loc: float = 1.0,
                            max_size: Optional[int] = None):
    """
    Get the coordinate from between 0 and 1 of an item given its index in a
    collection.

    :param index: The 0-based index of the item in its collection.
    :param size: The number of items in the item's collection.
    :param max_loc: The maximum location of an item.
    :param max_size: The maximum number of items that can appear in any
                     collection.
                     Use if all items in all collections should be equally
                     spaced.
                     Leave as None to give each collection its own spacing.
    """
    if max_size is None:
        max_size = size
    spacing = max_loc / (max_size - 1 if max_size > 1 else 1)
    min_loc = (max_loc / 2) - ((size - 1) * spacing) / 2
    return min_loc + index * spacing
