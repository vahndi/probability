from typing import (
    Any, Callable, List, Optional, Protocol, TypeVar, runtime_checkable
)

from probability.calculations.calculation_context import CalculationContext
from probability.custom_types.calculation_types import CalculationValue
from probability.distributions.protocols import NUM_SAMPLES_COMPARISON


SelfCalcT = TypeVar('SelfCalcT', bound='SupportsCalculation')


@runtime_checkable
class SupportsOperator(Protocol):
    """
    Protocol for unary, binary, and aggregator calculation operators.
    """
    get_name: Callable[..., str]
    operate: Callable[..., Any]


class OperatorMixin(object):

    get_name: Callable[..., str]  # name of the result
    operate: Callable  # carry out the operation


@runtime_checkable
class SupportsCalculation(Protocol):
    """
    Protocol for probability calculations with a shared CalculationContext.
    """
    context: CalculationContext

    @property
    def input_calcs(self) -> List['SupportsCalculation']: ...

    @property
    def name(self) -> str: ...

    def set_context(
            self: SelfCalcT, context: CalculationContext
    ) -> SelfCalcT: ...

    def output(
            self, num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> CalculationValue: ...

    def rvs(
            self, num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> CalculationValue: ...


class ProbabilityCalculationMixin(object):

    context: CalculationContext
    set_context: Callable[[Any], 'ProbabilityCalculationMixin']

    @property
    def input_calcs(self) -> List['ProbabilityCalculationMixin']:
        """
        Return the inputs to the calculation.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Return the name of the calculation.
        """
        raise NotImplementedError

    def output(
            self,
            num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> 'CalculationValue':
        """
        Return the existing output if it exists,
        otherwise sample it from scratch.

        :param num_samples: Number of samples to draw.
        """
        raise NotImplementedError

    def rvs(
            self,
            num_samples: Optional[int] = NUM_SAMPLES_COMPARISON
    ) -> 'CalculationValue':
        """
        Resample the calculation output whether or not it exists.

        :param num_samples: Number of samples to draw.
        """
        self.context.clear()
        return self.output(num_samples)

