"""
A Catalog of Noninformative Priors
Ruoyong Yang
Parexel International
 Revere Drive Suite
Northbrook IL


ruoyongyangparexelcom

James O Berger
ISDS
Duke University
Duahrm NC
bergerstatdukeedu

December 1998
"""


class UniformPrior(object):
    """
    By this, we just mean the constant density, with the constant typically
    chosen to be 1 (unless the constant can be chosen to yield a proper
    density). This choice was, of course, popularized by Laplace (1812).
    """

    class Binomial(object):

        alpha = 1
        beta = 1

    class Geometric(object):
        """
        Not explicitly listed in reference but satisfies constant density
        requirement for any compound distribution with a Beta prior.
        """
        alpha = 1
        beta = 1

    class Dirichlet(object):

        alpha = 1


class JeffreysPrior(object):
    """
    This is defined as

        π(θ) = sqrt(det[I(θ)])

    where I(θ) is the Fisher information matrix.

    This was proposed in Jeffreys (1961), as a solution to the problem that the
    uniform prior does not yield and analysis invariant to choice of
    parameterization. Note that, in specific situations, Jeffreys often
    recommended non-informative priors that differed from the formal Jeffreys
    prior.
    """

    class Binomial(object):

        alpha = 0.5
        beta = 0.5


class ReferencePrior(object):
    """
    This approach was developed in Bernardo (1979), and modified for
    multi-parameter problems in Berger and Bernardo (1992c). The approach cannot
    be simply described, but it can be roughly thought of as trying to modify
    the Jeffreys prior by reducing the dependence among parameters that is
    frequently induced by the Jeffreys prior; there are many well-known examples
    in which the Jeffreys prior yields poor performance (even inconsistency)
    because of this dependence.
    """

    class Binomial(object):

        alpha = 0.5
        beta = 0.5


class MDIPrior(object):
    """
    The Maximal Data Information Prior (MDIP): This approach was developed in
    Zellner (1971), based on an information argument. It is given by

        π(θ) = exp(integral[p(x|θ) log p(x|θ) dx])

    where p(x|θ) is the data density function.
    """
    pass


class VaguePrior(object):

    class Gamma(object):
        """
        https://math.stackexchange.com/a/456050
        """
        alpha = 0.001
        beta = 0.001


class ImproperPrior(object):

    class Gamma(object):
        """
        https://math.stackexchange.com/a/456050
        """
        alpha = 0.0
        beta = 0.0
