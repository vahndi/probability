from numpy import array
from scipy.stats import beta as beta_scipy

from probability.distributions.mixins.rv_continuous_mixin import RVContinuousMixin


class Beta(RVContinuousMixin):

    def __init__(self, alpha: float, beta: float):

        self._alpha = alpha
        self._beta = beta
        self._distribution = beta_scipy(alpha, beta)
        self._label = f'Beta(α={self._alpha}, β={self._beta})'

    def __str__(self):

        return f'Beta(α={self._alpha}, β={self._beta})'

    def __repr__(self):

        return f'Beta(alpha={self._alpha}, beta={self._beta})'


b = Beta(3, 1)

pdf_1 = b.pdf.at(0.5)
print(pdf_1)
pdf_2 = b.pdf.at(array([0.5, 0.6, 0.7]))
print(pdf_2)
b1 = Beta(1, 2)
b2 = Beta(3, 4)
