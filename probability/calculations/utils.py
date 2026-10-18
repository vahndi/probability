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

    def _sync_single(calc: ProbabilityCalculationMixin):
        if hasattr(calc, 'set_context'):
            calc.set_context(context)
        else:
            calc.context = context
            for calc_input in calc.input_calcs:
                _sync_single(calc_input)

    for calculation in calculations:
        if isinstance(calculation, ProbabilityCalculationMixin):
            _sync_single(calculation)
        elif isinstance(calculation, Series):
            for key, value in calculation.items():
                if isinstance(value, ProbabilityCalculationMixin):
                    _sync_single(value)
        elif isinstance(calculation, DataFrame):
            for ix, col in product(calculation.index, calculation.columns):
                value = calculation.loc[ix, col]
                if isinstance(value, ProbabilityCalculationMixin):
                    _sync_single(value)
        else:
            raise TypeError(f'Cannot sync context for type {type(calculation)}')