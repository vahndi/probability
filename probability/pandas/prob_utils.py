from pandas import merge, Series, DataFrame
from typing import Any, Tuple, List

"""
The following methods assume that the distribution is represented as follows.
- The distribution is a pandas Series.
- The columns of the index are named after the variables in the distribution.
- The rows of the the index contain each unique combination of values of the variables in the distribution.
- The name of the Series is 'p'.
- The values of the Series represent the probability of the combination of variable values in the associated index row.
"""


def margin(distribution: Series, *margins) -> Series:
    """
    Marginalize the distribution over the variables not in args, leaving the marginal probability of args.

    :param distribution: The probability distribution to marginalize e.g. P(A,B,C,D).
    :param margins: Names of variables to put in the margin e.g. 'C', 'D'.
    :return: P(C,D)
    """
    return distribution.to_frame().groupby(list(margins))['p'].sum()


_match_codes: List[str] = ['eq', 'ne', 'lt', 'gt', 'le', 'ge', 'in', 'not_in']
comparator_symbols = {
    'eq': lambda arg, val: f'{arg}={val}',
    'ne': lambda arg, val: f'{arg}≠{val}',
    'lt': lambda arg, val: f'{arg}<{val}',
    'gt': lambda arg, val: f'{arg}>{val}',
    'le': lambda arg, val: f'{arg}≤{val}',
    'ge': lambda arg, val: f'{arg}≥{val}',
    'in': lambda arg, vals: '{}∈{}'.format(
        arg, '{' + ",".join([str(val) for val in vals]) + '}'
    ),
    'not_in': lambda arg, vals: '{}∉{}'.format(
        arg, '{' + ",".join([str(val) for val in vals]) + '}'
    )
}


def valid_name_comparator(name_comparator: str, var_names: List[str]) -> bool:
    """
    Return whether the given name is a valid conditioning filter name for any of the variables in var_names.

    :param name_comparator: Amalgamation of variable name and filtering comparator in the form '{name}__{comparator}'.
    :param var_names: List of valid variables names to look for in `name_comparator`.
    """
    for var_name in var_names:
        if name_comparator == var_name:
            return True
    for var_name in var_names:
        for code in _match_codes:
            if name_comparator == var_name + '__' + code:
                return True
    return False


def _filter_distribution(distribution: DataFrame, name_comparator: str, value: Any) -> Tuple[DataFrame, str]:
    """
    Filter probability distribution data using the variable name, comparator code and value.

    :param distribution: The probability distribution data to filter.
    :param name_comparator: Amalgamation of variable name and filtering comparator in the form '{name}__{comparator}'.
    :param value: Value to filter to.
    :return: Filtered Data, Variable Name
    """
    var_names = [col for col in distribution if col != 'p']

    def match_var(code: str) -> bool:
        return name_comparator in [f'{var_name}__{code}' for var_name in var_names]

    if name_comparator in var_names:
        return distribution.loc[distribution[name_comparator] == value], name_comparator
    elif match_var('eq'):
        return distribution.loc[distribution[name_comparator[: -4]] == value], name_comparator[: -4]
    elif match_var('ne'):
        return distribution.loc[distribution[name_comparator[: -4]] != value], name_comparator[: -4]
    elif match_var('lt'):
        return distribution.loc[distribution[name_comparator[: -4]] < value], name_comparator[: -4]
    elif match_var('gt'):
        return distribution.loc[distribution[name_comparator[: -4]] > value], name_comparator[: -4]
    elif match_var('le'):
        return distribution.loc[distribution[name_comparator[: -4]] <= value], name_comparator[: -4]
    elif match_var('ge'):
        return distribution.loc[distribution[name_comparator[: -4]] >= value], name_comparator[: -4]
    elif match_var('in'):
        return distribution.loc[distribution[name_comparator[: -4]].isin(value)], name_comparator[: -4]
    elif match_var('not_in'):
        return distribution.loc[~distribution[name_comparator[: -8]].isin(value)], name_comparator[: -8]


def cond_name(name_comparator: str, var_names: List[str]) -> str:
    """
    Return the conditioning variable name from the name and comparator code.

    :param name_comparator: Amalgamation of variable name and filtering comparator in the form '{name}__{comparator}'.
    :param var_names: List of valid variables names to look for in `name_comparator`.
    """
    if name_comparator in var_names:
        return name_comparator
    for var_name in var_names:
        for code in _match_codes:
            if var_name + '__' + code == name_comparator:
                return var_name


def cond_name_and_symbol(name_comparator: str, value, var_names: List[str]) -> str:
    """
    Return the conditioning variable name and mathematical symbol from the name and comparator code.

    :param name_comparator: Amalgamation of variable name and filtering comparator in the form '{name}__{comparator}'.
    :param var_names: List of valid variables names to look for in `name_comparator`.
    :param value: Value to filter to.
    """
    for var_name in var_names:
        for code in _match_codes:
            if var_name + '__' + code == name_comparator:
                return comparator_symbols[code](var_name, value)
    if name_comparator in var_names:
        return f'{name_comparator}={value}'


def condition(distribution: Series, *cond_vars) -> Series:
    """
    Condition the distribution on given and/or not-given values of the variables.

    :param distribution: The probability distribution to condition e.g. P(A,B,C,D).
    :param cond_vars: Names of variables to condition over every value of e.g. 'C'.
    :return: Conditioned distribution. Filtered to only given values of the cond_values.
             Contains a stacked Series of probabily distributions for each combination of conditioning variable values.
             e.g. P(A,B|C,D=d1), P(A,B|C,D=d2) etc.
    """
    col_names = distribution.index.names
    var_names = ([
        n for n in col_names
        if n not in cond_vars
    ])
    var_names.extend([n for n in col_names if n in cond_vars])
    data = distribution.copy().reset_index()
    not_given_vars = list(cond_vars)
    if not_given_vars:
        # find total probabilities for each combination of unique values in the conditional variables e.g. P(C)
        sums = data.groupby(not_given_vars).sum().reset_index()
        # normalize each individual probability e.g. p(Ai,Bj,Ck,Dl) to probability of its conditional values p(Ck)
        sums = sums[not_given_vars + ['p']].rename(columns={'p': 'p_sum'})
        merged = merge(left=data, right=sums, on=not_given_vars)
        merged['p'] = merged['p'] / merged['p_sum']
        data = merged[var_names + ['p']]
    return data.set_index(var_names)['p']


def given(distribution: Series, **givens) -> Series:
    """
    Condition the distribution on given and/or not-given values of the variables.

    :param distribution: The probability distribution to condition e.g. P(A,B,C,D).
    :param givens: Names and values of variables to condition on a given value e.g. D=1.
    :return: Conditioned distribution. Filtered to only given values of the cond_values.
             Contains a single probability distribution summing to 1.
    """
    col_names = distribution.index.names
    var_names = ([
        n for n in col_names
        if n not in givens.keys()
        and not set([f'{n}__{code}' for code in _match_codes]).intersection(givens.keys())
    ])
    data = distribution.copy().reset_index()
    for given_var, given_val in givens.items():
        # filter individual probabilities to given values e.g. P(A,B,C,D=d1)
        data, var_name = _filter_distribution(data, given_var, given_val)
        var_names.append(var_name)
    # normalize each individual remaining probability P(Ai,Bj,Ck,d1)
    # to the sum of remaining probabilities P(A,B,C,d1)
    data['p'] = data['p'] / data['p'].sum()
    return data.set_index([var_name for var_name in var_names if var_name not in givens.keys()])['p']


def multiply(conditional: Series, marginal: Series) -> Series:

    # P(a,b) = P(a|b) * P(b)
    # joint = conditional * marginal
    marginal_vars = marginal.index.names
    non_marginal_vars = [v for v in conditional.index.names if v not in marginal_vars]
    cond_data = conditional.copy().rename('p_cond').to_frame().reset_index()
    joint_data = marginal.copy().rename('p_joint').to_frame().reset_index()
    merged = merge(left=cond_data, right=joint_data, on=marginal_vars)
    merged['p'] = merged['p_cond'] * merged['p_joint']
    results = merged.groupby(non_marginal_vars + marginal_vars)['p'].sum()
    return results
