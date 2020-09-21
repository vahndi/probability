from itertools import product
from typing import Union

from pandas import Series, DataFrame

from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.custom_types.calculation_types import CalculationValue


class CalculationContext(object):

    def __init__(self):

        self._context = {}

    def __setitem__(self, name: str, value: CalculationValue):

        self._context[name] = value

    def __getitem__(self, name: str) -> CalculationValue:

        return self._context[name]

    def context(self) -> dict:

        return self._context

    def has_object_named(self, name: str) -> bool:

        return name in self._context.keys()


def sync_context(
        *calculations: Union[ProbabilityCalculationMixin,
                             Series, DataFrame]
):

    context = CalculationContext()
    for calculation in calculations:
        if isinstance(calculation, ProbabilityCalculationMixin):
            calculation.context = context
            for calc_input in calculation.input_calcs:
                calc_input.context = context
        elif isinstance(calculation, Series):
            for key, value in calculation.items():
                if isinstance(value, ProbabilityCalculationMixin):
                    value.context = context
                    for calc_input in value.input_calcs:
                        calc_input.context = context
        elif isinstance(calculation, DataFrame):
            for ix, col in product(calculation.index, calculation.columns):
                value = calculation.loc[ix, col]
                if isinstance(value, ProbabilityCalculationMixin):
                    value.context = context
                    for calc_input in value.input_calcs:
                        calc_input.context = context
        else:
            raise TypeError(f'Cannot sync context for type {type(calculation)}')
