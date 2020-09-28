from operator import sub

from probability.calculations.operators.binary_operator import BinaryOperator


class Subtract(
    BinaryOperator,
    object
):

    symbol = '-'
    operator = sub
    pandas_op_name = 'sub'
