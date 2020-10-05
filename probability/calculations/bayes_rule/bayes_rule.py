from typing import Union

from probability.custom_types.external_custom_types import AnyFloatMap
from probability.custom_types.internal_custom_types import AnyBetaMap, \
    AnyDirichletMap
from probability.distributions import Beta, Dirichlet


class BayesRule(object):

    _prior: Union[float, Beta, AnyFloatMap, Dirichlet]
    _likelihood: Union[float, AnyFloatMap, AnyBetaMap, AnyDirichletMap]

    @property
    def prior(self) -> Union[float, Beta, AnyFloatMap, Dirichlet]:

        return self._prior

    @property
    def likelihood(self) -> Union[
            float, AnyFloatMap, AnyBetaMap, AnyDirichletMap
    ]:

        return self._likelihood
