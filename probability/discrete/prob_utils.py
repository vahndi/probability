from typing import Any, Tuple, List

from pandas import Series, DataFrame


def _filter_distribution(
        distribution: DataFrame,
        distribution_name: str,
        name_comparator: str, value: Any
) -> Tuple[DataFrame, str]:
    """
    Filter probability distribution data using the variable name, comparator
    code and value.

    :param distribution: The probability distribution data to filter.
    :param name_comparator: Amalgamation of variable name and filtering
                            comparator in the form '{name}__{comparator}'.
    :param value: Value to filter to.
    :return: Filtered Data, Variable Name
    """
    var_names = [col for col in distribution if col != distribution_name]

    def match_var(code: str) -> bool:
        return name_comparator in [f'{var_name}__{code}'
                                   for var_name in var_names]

    if name_comparator in var_names:
        return distribution.loc[
            distribution[name_comparator] == value
        ], name_comparator
    elif match_var('eq'):
        return distribution.loc[
            distribution[name_comparator[: -4]] == value
        ], name_comparator[: -4]
    elif match_var('ne'):
        return distribution.loc[
            distribution[name_comparator[: -4]] != value
        ], name_comparator[: -4]
    elif match_var('lt'):
        return distribution.loc[
            distribution[name_comparator[: -4]] < value
        ], name_comparator[: -4]
    elif match_var('gt'):
        return distribution.loc[
            distribution[name_comparator[: -4]] > value
        ], name_comparator[: -4]
    elif match_var('le'):
        return distribution.loc[
            distribution[name_comparator[: -4]] <= value
        ], name_comparator[: -4]
    elif match_var('ge'):
        return distribution.loc[
            distribution[name_comparator[: -4]] >= value
        ], name_comparator[: -4]
    elif match_var('in'):
        return distribution.loc[
            distribution[name_comparator[: -4]].isin(value)
        ], name_comparator[: -4]
    elif match_var('not_in'):
        return distribution.loc[
            ~distribution[name_comparator[: -8]].isin(value)
        ], name_comparator[: -8]


def p(distribution: Series, **joint_vars_vals) -> float:
    """
    Calculate the probability of the values of the joint values given.

    :param distribution: Distribution data to calculate probability from.
    :param joint_vars_vals: Names and values of variables to find probability of
                            e.g. `C=1`, `D__le=1`.
    """
    dist_name = distribution.name
    data = distribution.copy().reset_index()
    for joint_var, joint_val in joint_vars_vals.items():
        # filter individual probabilities to specified values e.g. P(A,B,C,D=d1)
        data, var_name = _filter_distribution(
            data, dist_name, joint_var, joint_val
        )
    # calculate probability
    return data[dist_name].sum()


def given(distribution: Series, **givens) -> Series:
    """
    Condition the distribution on given and/or not-given values of the
    variables.

    :param distribution: The probability distribution to condition
                         e.g. P(A,B,C,D).
    :param givens: Names and values of variables to condition on a given value
                   e.g. D=1.
    :return: Conditioned distribution. Filtered to only given values of the
             cond_values.
             Contains a single probability distribution summing to 1.
    """
    dist_name = distribution.name
    col_names = distribution.index.names
    joint_names = ([
        n for n in col_names
        if n not in givens.keys()  # not a given variable name w/o comparator
        and not set(
            [f'{n}__{code}' for code in _match_codes]
        ).intersection(givens.keys())  # not a given variable name w/ comparator
    ])
    var_names = joint_names.copy()
    data = distribution.copy().reset_index()
    for given_var, given_val in givens.items():
        # filter individual probabilities to given values e.g. P(A,B,C,D=d1)
        data, var_name = _filter_distribution(
            data, dist_name, given_var, given_val
        )
        var_names.append(var_name)
    # normalize each individual remaining probability P(Ai,Bj,Ck,d1)
    # to the sum of remaining probabilities P(A,B,C,d1)
    data[dist_name] = data[dist_name] / data[dist_name].sum()
    return data.set_index([
        var_name for var_name in var_names
        if var_name not in givens.keys()
    ])[dist_name]


_match_codes: List[str] = ['eq', 'ne', 'lt', 'gt', 'le', 'ge', 'in', 'not_in']


def valid_name_comparator(name_comparator: str, var_names: List[str]) -> bool:
    """
    Return whether the given name is a valid conditioning filter name for any of
    the variables in var_names.

    :param name_comparator: Amalgamation of variable name and filtering
                            comparator in the form '{name}__{comparator}'.
    :param var_names: List of valid variables names to look for in
                      `name_comparator`.
    """
    for var_name in var_names:
        if name_comparator == var_name:
            return True
    for var_name in var_names:
        for code in _match_codes:
            if name_comparator == var_name + '__' + code:
                return True
    return False