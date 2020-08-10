import matplotlib.pyplot as plt

from probability.distributions.discrete.negative_binomial import NegativeBinomial


k_wiki = list(range(26))


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Negative_binomial_distribution#/media/File:Negbinomial.gif
    """
    r_s = [1, 2, 3, 4, 5, 10, 20, 40]
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 9))
    for r in r_s:
        ax = axes.flat[r_s.index(r)]
        NegativeBinomial(r=r, p=0.5).with_label(f'r = {r}').plot(
            k=k_wiki, kind='line', color='blue', ax=ax,
            ls='-', marker='o', vlines=True
        )
        ax.set_ylim(0, 0.12)
        ax.legend(loc='upper right')
    fig.suptitle('Probability mass function')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
