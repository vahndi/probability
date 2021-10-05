from operator import mul

from probability.calculations.operators.binary_operators.binary_operator \
    import BinaryOperator


class MultiplyOperator(
    BinaryOperator,
    object
):
    """
    Operator to multiply the values of 2 distributions.
    """
    symbol = '*'
    operator = mul
    pandas_op_name = 'mul'
