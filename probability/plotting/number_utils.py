from typing import Union


Scalar = Union[int, float]


def format_as_percent(number: Scalar, ndp: int = 1) -> str:
    """
    Format a proportion as a percentage with a given number of decimal places.

    :param number: The number to format.
    :param ndp: Number of decimal places.
    """
    return f'{number:.{ndp}%}'


def format_as_integer(number: Scalar, kmbt: bool = False) -> str:
    """
    Format a number as an integer, with comma separators and an optional suffix
    of K, M, B or T for large numbers.

    :param number: The number to format.
    :param kmbt: Whether to abbreviate numbers using K, M, B and T for
                 thousands, millions, billions and trillions.
    """
    if not kmbt:
        return f'{int(number):,}'
    for power, abbrev in zip([12, 9, 6, 3], ['T', 'B', 'M', 'K']):
        if number >= 10 ** power:
            num = number / 10 ** power
            if num == int(num):
                num = int(num)
            return f'{num:,}{abbrev}'
    if number == int(number):
        number = int(number)
    return f'{number:,}'
