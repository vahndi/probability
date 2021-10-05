from probability.calculations.mixins import ProbabilityCalculationMixin


class SimpleCalculation(
    ProbabilityCalculationMixin,
    object
):
    """
    Base class for SampleCalculation and ValueCalculation.

    Used for type-checking
    """
    pass
