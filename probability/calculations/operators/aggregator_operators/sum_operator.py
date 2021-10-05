from pandas import DataFrame, Series

from probability.calculations.mixins import OperatorMixin


class SumOperator(
    OperatorMixin,
    object
):
    """
    Operator to sum a single variable.
    """
    @classmethod
    def get_name(cls, name: str):
        """
        Return the name of the result.

        :param name: Name of the input.
        """
        return f'sum({name})'

    @classmethod
    def operate(cls, value: DataFrame) -> Series:
        """
        Apply the sum operation across a DataFrame's columns.

        :param value: The DataFrame of distributions to sum across.
        """
        if isinstance(value, DataFrame):
            result = value.sum(axis=1)
            names_csv = ', '.join(value.columns.to_list())
            result.name = f'sum({names_csv})'
            return result
        else:
            raise TypeError('value for Sum aggregator must be DataFrame')
