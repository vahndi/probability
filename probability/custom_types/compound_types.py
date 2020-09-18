from typing import Union

from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin

RVMixin = Union[RVContinuous1dMixin, RVDiscrete1dMixin]