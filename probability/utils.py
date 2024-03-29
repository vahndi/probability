from itertools import permutations
from typing import List, Tuple

from pandas import Series


def k_tuples_summing_to_n(k, n) -> List[Tuple[int]]:
    """
    Return a list of k-tuples of values that sum to n.

    :param k: The number of items in each tuple.
    :param n: The

    Calculation time increases exponentially with k
    - not recommended to use for k > 3 if n >> k.
    """
    def valid(val):
        return sum(val) == n

    return list(filter(valid, list(permutations(range(n + 1), k))))


def all_are_none(*args) -> bool:
    """
    Return True if every arg is None.
    """
    return all([arg is None for arg in args])


def none_are_none(*args) -> bool:
    """
    Return True if no arg is None.
    """
    return not any([arg is None for arg in args])


def any_are_not_none(*args) -> bool:
    """
    Return True if any arg is not None.
    """
    return any([arg is not None for arg in args])


def any_are_none(*args) -> bool:
    """
    Return True if any arg is None.
    """
    return any([arg is None for arg in args])


def one_is_none(*args) -> bool:
    """
    Return True if exactly one arg is None.
    """
    return sum([arg is None for arg in args]) == 1


def series_is_binary(data: Series) -> bool:
    """
    Return True if every value of the Series is 0 or 1.
    """
    return (
        set(data.unique()).issubset({0, 1})
    )


def is_scalar(arg) -> bool:
    """
    Return True if the arg is a scalar.
    """
    return isinstance(arg, int) or isinstance(arg, float)


def num_format(number: float, max_dp: int) -> str:

    for ndp in range(max_dp):
        if round(number, ndp) == number:
            return f'{number:0.{ndp}f}'
    return f'{number:0.{max_dp}f}'
