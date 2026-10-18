import matplotlib.pyplot as plt
from numpy.ma import arange
from pandas import Series

from probability.distributions import Beta
from probability.distributions.conjugate.beta_binomial_conjugate import \
    BetaBinomialConjugate
from probability.plotting import new_subplots


k = range(11)
x = arange(0, 1, 0.01)


def plot_ml_app():
    """
    Machine Learning: A Probabilistic Perspective. Figure 3.6
    """
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    bb_1 = BetaBinomialConjugate(alpha=2, beta=2, n=20, k=3)
    bb_1.prior().plot(x=x, color='red', ax=ax[0])
    Beta(alpha=3, beta=17).prepend_to_label('Likelihood: ').plot(
        x=x, color='black', ax=ax[0]
    )  # using a Beta to plot likelihood on the same scale
    bb_1.posterior().plot(x=x, color='blue', ax=ax[0])
    ax[0].legend()
    bb_2 = BetaBinomialConjugate(alpha=5, beta=2, n=20, k=11)
    bb_2.prior().plot(x=x, color='red', ax=ax[1])
    Beta(alpha=11, beta=13).prepend_to_label('Likelihood: ').plot(
        x=x, color='black', ax=ax[1]
    )  # using a Beta to plot likelihood on the same scale
    bb_2.posterior().plot(x=x, color='blue', ax=ax[1])
    ax[1].legend()
    plt.show()


def plot_examples():

    alphas = [0.5, 5, 1, 2, 2]
    betas = [0.5, 1, 3, 2, 5]
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    k_data = 7
    n_data = 10
    n_predict = 15
    k_predict = range(16)
    _, axes = new_subplots(n_rows=2, n_cols=3)
    (
        ax_prior, ax_data, ax_posterior,
        ax_prior_predictive, ax_likelihood, ax_posterior_predictive
    ) = axes.flat

    for alpha, beta, color in zip(alphas, betas, colors):
        bb = BetaBinomialConjugate(alpha=alpha, beta=beta, n=n_data, k=k_data)
        bb.prior().plot(x=x, color=color, ax=ax_prior)
        bb.posterior().plot(x=x, color=color, ax=ax_posterior)
        bb.prior_predictive(n_=n_predict).plot(
            k=k_predict, color=color, kind='line',
            marker='o', ax=ax_prior_predictive
        )
        bb.posterior_predictive(n_=n_predict).plot(
            k=k_predict, color=color, kind='line',
            marker='o', ax=ax_posterior_predictive
        )

    ax_prior.set_title('prior')
    ax_prior.legend()
    ax_posterior.set_title('posterior')
    ax_posterior.legend()
    ax_prior_predictive.set_title('prior predictive')
    ax_prior_predictive.legend()
    ax_posterior_predictive.set_title('posterior predictive')
    ax_posterior_predictive.legend()
    # plot data
    observations = Series(
        index=range(1, 11),
        data=[1] * 7 + [0] * 3).sample(frac=1)
    observations.index = range(1, 11)
    observations.plot.bar(ax=ax_data, color='k')
    # plot likelihood
    bb.likelihood().plot(k=k, color='k', ax=ax_likelihood)
    ax_likelihood.set_title('likelihood')
    ax_likelihood.legend()
    ax_data.set_title('data')
    ax_data.set_xlabel('i')
    ax_data.set_ylabel('$X_i$')
    plt.show()


def plot_example():

    bb = BetaBinomialConjugate(alpha=2, beta=5, n=10, k=7)
    bb.plot(n_=15)
    plt.show()


def plot_comparison():

    bb_10 = BetaBinomialConjugate(n=10, k=7)
    bb_100 = BetaBinomialConjugate(n=100, k=70)
    _, axes = new_subplots(n_rows=1, n_cols=2)
    bb_10.posterior_predictive(n_=100).plot(k=range(101), ax=axes[0])
    bb_100.posterior_predictive(n_=100).plot(k=range(101), ax=axes[1])
    plt.show()


if __name__ == '__main__':

    plot_ml_app()
    plot_examples()
    plot_example()
    plot_comparison()
