from operator import truediv

from probability.calculations.operators.binary_operator import BinaryOperator


class Divide(
    BinaryOperator,
    object
):

    symbol = '/'
    operator = truediv
    pandas_op_name = 'div'
