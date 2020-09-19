from probability.calculations.calculation import CalculationInput
from probability.calculations.mixins import OperatorMixin


class BinaryOperator(OperatorMixin):

    symbol: str

    @classmethod
    def get_name(cls, name_1, name_2):

        return f'{name_1} {cls.symbol} {name_2}'


class AddDistributions(
    BinaryOperator,
    object
):

    symbol = '+'

    @staticmethod
    def operate(input_1: CalculationInput,
                input_2: CalculationInput,
                num_samples: int = None):
        raise NotImplementedError


class SubtractDistributions(
    BinaryOperator,
    object
):

    symbol = '-'

    @staticmethod
    def operate(input_1: CalculationInput,
                input_2: CalculationInput,
                num_samples: int = None):
        raise NotImplementedError


class MultiplyDistributions(
    BinaryOperator,
    object
):

    symbol = '*'

    @staticmethod
    def operate(value_1, value_2):

        return value_1 * value_2


class DivideDistributions(
    BinaryOperator,
    object
):

    symbol = '/'

    @staticmethod
    def operate(input_1: CalculationInput,
                input_2: CalculationInput,
                num_samples: int = None):
        raise NotImplementedError


class ComplementDistribution(
    OperatorMixin,
    object
):

    @classmethod
    def get_name(cls, name):
        return f'1 - {name}'

    @staticmethod
    def operate(value):

        return 1 - value

