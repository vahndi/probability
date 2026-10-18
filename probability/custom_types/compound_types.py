from typing import Union

from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin
from probability.distributions.protocols import SupportsRV1d

RVMixin = Union[RVContinuous1dMixin, RVDiscrete1dMixin, SupportsRV1d]

