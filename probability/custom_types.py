from typing import Union

from probability.distributions.mixins.rv_continuous_mixin import RVContinuousMixin
from probability.distributions.mixins.rv_discrete_mixin import RVDiscreteMixin


RVMixin = Union[RVContinuousMixin, RVDiscreteMixin]
