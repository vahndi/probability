import matplotlib.pyplot as plt
from numpy import arange, logspace, linspace
from probability.distributions.conjugate.gamma_exponential import GammaExponential
from probability.plots import new_axes


rates = arange(0, 5, 0.05)
durations = linspace(0.1, 10, 1000)
print(durations)


def plot_gamma_exp():

    ge = GammaExponential(alpha=10, beta=5, n=100, x_mean=0.3)
    ax = new_axes()
    ge.prior().plot(x=rates, color='r', ax=ax)
    ge.posterior().plot(x=rates, color='g', ax=ax)
    plt.show()
    ax = new_axes()
    p_x = ge.predict_proba(durations)
    p_x.plot(ax=ax)
    plt.show()
    return ge


ge = plot_gamma_exp()

