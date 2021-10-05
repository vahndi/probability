from typing import Callable, Any, List

from pandas import Series, concat

from probability.calculations.mixins import OperatorMixin
from probability.custom_types.calculation_types import CalculationValue
from probability.utils import is_scalar


class ArrayOperator(OperatorMixin, object):
    """
    An Operator that produces an output from a list of inputs
    e.g. Minimum, Maximum, Mean, Median or some other array method.
    """
    symbol: str
    operator: Callable[[Any], Any]
    pandas_op_name: str

    @classmethod
    def get_name(cls, names: List[str]) -> str:
        """
        Return the name of the Operator applied to a list of inputs.

        :param names: List of input names.
        """
        cs_names = ', '.join(names)
        return f'{cls.symbol}({cs_names})'

    @classmethod
    def operate(
            cls,
            values: List[CalculationValue]
    ) -> CalculationValue:
        """
        Execute the operation on a list of input values.

        :param values: The values to operate on.
        """
        if all([is_scalar(v) for v in values]):
            return cls.operator(values)
        elif all([isinstance(v, Series) for v in values]):
            data = concat(values, axis=1)
            return data.apply(cls.pandas_op_name, axis=1)
        else:
            types = [type(v) for v in values]
            raise TypeError(f'Unsupported types in values. {types}')
