from operator import sub

from probability.calculations.operators.binary_operators.binary_operator \
    import BinaryOperator


class SubtractOperator(
    BinaryOperator,
    object
):
    """
    Operator to subtract the values of 2 distributions.
    """
    symbol = '-'
    operator = sub
    pandas_op_name = 'sub'
