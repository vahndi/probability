from probability.calculations.operators.binary_operator import BinaryOperator


class Divide(
    BinaryOperator,
    object
):

    symbol = '/'

    @staticmethod
    def operate(value_1, value_2):
        raise NotImplementedError