from probability.calculations.mixins import OperatorMixin


class BinaryOperator(OperatorMixin):

    symbol: str

    @classmethod
    def get_name(cls, name_1: str, name_2: str):

        return f'{name_1} {cls.symbol} {name_2}'


class Add(
    BinaryOperator,
    object
):

    symbol = '+'

    @staticmethod
    def operate(value_1, value_2,
                num_samples: int = None):
        raise NotImplementedError


class Subtract(
    BinaryOperator,
    object
):

    symbol = '-'

    @staticmethod
    def operate(value_1, value_2,
                num_samples: int = None):
        raise NotImplementedError


class Multiply(
    BinaryOperator,
    object
):

    symbol = '*'

    @staticmethod
    def operate(value_1, value_2):

        return value_1 * value_2


class Divide(
    BinaryOperator,
    object
):

    symbol = '/'

    @staticmethod
    def operate(value_1, value_2):
        raise NotImplementedError


class Complement(
    OperatorMixin,
    object
):

    @classmethod
    def get_name(cls, name: str):
        return f'1 - {name}'

    @staticmethod
    def operate(value):

        return 1 - value
