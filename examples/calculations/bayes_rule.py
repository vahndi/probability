import matplotlib.pyplot as plt
from pandas import concat, Series, DataFrame
from seaborn import boxplot

from probability.calculations.bayes_rule.binary_bayes_rule import \
    BinaryBayesRule
from probability.distributions import Poisson
from probability.plotting import new_axes

"""
wins ∝ P($|R=win)
losses ∝ P($|R=loss)
"""


wins: Series = \
    300 * Poisson(lambda_=8).pmf().at(range(1, 51)).rename('win')
wins.index = range(100_000, 5_000_001, 100_000)
wins = wins.round(0).astype(int)
losses: Series = \
    700 * Poisson(lambda_=10).pmf().at(range(1, 51)).rename('loss')
losses.index = range(100_000, 5_000_001, 100_000)
losses = losses.round(0).astype(int)

data = concat([wins, losses], axis=1)[['loss', 'win']]
ax = new_axes()
data.plot.bar(ax=ax)
plt.show()


def plot_probs(probs: DataFrame, weight):

    ax = new_axes()
    data = probs.stack(
        level=['likelihood', 'prior']
    ).rename('posterior').reset_index()
    boxplot(data=data, x='likelihood', y='posterior', hue='prior', ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim(0, 1.05)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_title(str(weight))
    plt.show()


for weight in (0, 0.001, 0.01, 0.1, 1.0, 1000):

    br = BinaryBayesRule.from_counts(data, prior_weight=weight)
    samples = br.posterior(10000)
    plot_probs(samples, weight)
