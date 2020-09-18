from typing import Mapping

from pandas import Series

from probability.distributions import Beta, Dirichlet


def is_mapping(instance) -> bool:

    return isinstance(instance, Mapping) or isinstance(instance, Series)


def all_values_are(mapping: Mapping, value_type: type) -> bool:

    if isinstance(mapping, dict):
        values = mapping.values()
    elif isinstance(mapping, Series):
        values = mapping.values
    else:
        raise TypeError('mapping is not a Series or dict')
    return all([isinstance(value, value_type) for value in values])


def is_any_float_map(instance) -> bool:

    return is_mapping(instance) and all_values_are(instance, float)


def is_any_beta_map(instance) -> bool:

    return is_mapping(instance) and all_values_are(instance, Beta)


def is_any_dirichlet_map(instance) -> bool:

    return is_mapping(instance) and all_values_are(instance, Dirichlet)
