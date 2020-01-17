from itertools import permutations
from typing import List, Tuple, Iterable


def k_tuples_summing_to_n(k, n) -> List[Tuple[int]]:
    """
    Return a list of k-tuples of values that sum to n.

    :param k: The number of items in each tuple.
    :param n: The

    Calculation time increases exponentially with k - not recommended to use for k > 3 if n >> k.
    """
    def valid(val):
        return sum(val) == n

    return list(filter(valid, list(permutations(range(n + 1), k))))


def all_are_none(*args) -> bool:
    return all([arg is None for arg in args])


def none_are_none(*args) -> bool:
    return not any([arg is None for arg in args])


def any_are_not_none(*args) -> bool:
    return any([arg is not None for arg in args])


def any_are_none(*args) -> bool:
    return any([arg is None for arg in args])
