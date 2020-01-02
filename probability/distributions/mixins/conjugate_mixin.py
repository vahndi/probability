from numpy import ndarray


class ConjugateMixin(object):

    # region component distributions

    def prior(self, support: ndarray = None):
        return

    def likelihood(self, support=None):
        return

    def posterior(self, support=None):
        return

    # endregion

    # region plots

    def plot_prior(self, support=None):
        return

    def plot_likelihood(self, support=None):
        return

    def plot_posterior(self, support=None):
        return

    # end region

    # region calcs

    def posterior_mean(self):
        return

    def posterior_hpd(self):
        return

    # end region
