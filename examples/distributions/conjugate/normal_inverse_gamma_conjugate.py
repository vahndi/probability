import matplotlib.pyplot as plt
import numpy as np

from probability.distributions.conjugate import NormalInverseGammaConjugate

# Generate some sample data with true mean = 5.0 and true std = 2.0 (variance = 4.0)
np.random.seed(42)
true_mu, true_sigma = 5.0, 2.0
x = np.random.normal(loc=true_mu, scale=true_sigma, size=50)

# Fit conjugate model
dist = NormalInverseGammaConjugate.infer_posterior(x)


def print_summary():
    print(dist)
    print("Posterior marginal μ:       ", dist.posterior_mu())
    print("Posterior marginal σ²:      ", dist.posterior_sigma_sq())
    print("Posterior predictive x_new: ", dist.posterior_predictive())
    print("95% HDI for μ:              ", dist.posterior_mu().hdi(0.95))
    print("95% HDI for σ²:             ", dist.posterior_sigma_sq().hdi(0.95))


def plot_grid():
    dist.plot()
    plt.show()


if __name__ == '__main__':
    print_summary()
    plot_grid()
