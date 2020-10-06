from typing import Mapping, Any

from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.distributions.continuous.beta import Beta
from probability.distributions.multivariate.dirichlet import Dirichlet

AnyBetaMap = Mapping[Any, Beta]
AnyDirichletMap = Mapping[Any, Dirichlet]
AnyCalculationMap = Mapping[Any, ProbabilityCalculationMixin]
