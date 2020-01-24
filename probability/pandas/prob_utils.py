from pandas import merge, Series

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


def condition(distribution: Series, *not_givens, **givens) -> Series:
    """
    Condition the distribution on given and/or not-given values of the variables.

    :param distribution: The probability distribution to condition e.g. P(A,B,C,D).
    :param not_givens: Names of variables to condition on every value e.g. 'C'.
    :param givens: Names and values of variables to condition on a given value e.g. D=1.
    :return: Conditioned distribution. Filtered to only given values of the givens.
             Contains a stacked Series of probabilities summing to 1 for each combination of not-given variable values.
             e.g. P(A,B|C,D=d1), P(A,B|C,D=d2) etc.
    """
    var_names = list(distribution.index.names)
    data = distribution.copy().reset_index()
    if givens:
        for given_var, given_val in givens.items():
            data = data.loc[data[given_var] == given_val]
        data['p'] = data['p'] / data['p'].sum()
    not_given_vars = list(not_givens)
    if not_given_vars:
        sums = data.groupby(not_given_vars).sum().reset_index()
        sums = sums[not_given_vars + ['p']].rename(columns={'p': 'p_sum'})
        merged = merge(left=data, right=sums, on=not_given_vars)
        merged['p'] = merged['p'] / merged['p_sum']
        data = merged[var_names + ['p']]
    return data.set_index(var_names)['p']
