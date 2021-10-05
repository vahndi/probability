from itertools import product
from typing import Union

from pandas import Series, DataFrame

from probability.calculations.calculation_context import CalculationContext
from probability.calculations.mixins import ProbabilityCalculationMixin


def sync_context(
        *calculations: Union[ProbabilityCalculationMixin,
                             Series, DataFrame]
):
    """
    Apply a new CalculationContext to Calculations, Series of Calculations,
    or DataFrames of Calculations.

    :param calculations: One or more Calculations, Series of Calculations,
                         or DataFrames of Calculations.
    """
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