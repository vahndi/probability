from operator import mul

from probability.calculations.operators.binary_operator import BinaryOperator


class Multiply(
    BinaryOperator,
    object
):

    symbol = '*'
    operator = mul
    pandas_op_name = 'mul'
