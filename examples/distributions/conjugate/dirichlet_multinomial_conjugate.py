from probability.distributions.conjugate.dirichlet_multinomial_conjugate import \
    DirichletMultinomialConjugate


def plot_example():

    dm = DirichletMultinomialConjugate(alpha=0, n=0, x=0)
    dm.plot()
