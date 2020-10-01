from pandas import DataFrame, Series

from probability.calculations.mixins import OperatorMixin


class Sum(
    OperatorMixin,
    object
):

    @classmethod
    def get_name(cls, name: str):

        return f'sum({name})'

    @classmethod
    def operate(cls, value: DataFrame) -> Series:

        if isinstance(value, DataFrame):
            result = value.sum(axis=1)
            names_csv = ', '.join(value.columns.to_list())
            result.name = f'sum({names_csv})'
            return result
        else:
            raise TypeError('value for Sum aggregator must be DataFrame')
