from mpl_format.axes import AxesFormatter
from pandas import concat, Series, DataFrame
from seaborn import boxplot

from probability.calculations.bayes_rule.binary_bayes_rule import \
    BinaryBayesRule
from probability.distributions import Poisson

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
axf = AxesFormatter()
data.plot.bar(ax=axf.axes)
axf.show()


def plot_probs(probs: DataFrame, weight):

    axf = AxesFormatter()
    data = probs.stack(
        level=['likelihood', 'prior']
    ).rename('posterior').reset_index()
    boxplot(data=data, x='likelihood', y='posterior', hue='prior')
    axf.rotate_x_tick_labels(90)
    axf.set_y_lim(0, 1.05)
    axf.set_axis_below().grid()
    axf.set_text(title=str(weight))
    axf.show()


for weight in (0, 0.001, 0.01, 0.1, 1.0, 1000):

    br = BinaryBayesRule.from_counts(data, prior_weight=weight)
    samples = br.posterior(10000)
    plot_probs(samples, weight)
