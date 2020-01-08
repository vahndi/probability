from probability.custom_types import RVMixin


class ConjugateMixin(object):

    # region component distributions

    def prior(self, **kwargs) -> RVMixin:
        raise NotImplementedError

    def likelihood(self, **kwargs) -> RVMixin:
        raise NotImplementedError

    def posterior(self, **kwargs) -> RVMixin:
        raise NotImplementedError

    # endregion

