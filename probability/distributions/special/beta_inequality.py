from math import lgamma

from numba import jit
from numpy import exp, sum, array


@jit(nopython=True)
def h(a, b, c, d):
    num = lgamma(a + c) + lgamma(b + d) + lgamma(a + b) + lgamma(c + d)
    den = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
    return exp(num - den)


@jit(nopython=True)
def g0(a, b, c):
    return exp(lgamma(a + b) + lgamma(a + c) - (lgamma(a + b + c) + lgamma(a)))


@jit(nopython=True)
def hiter(a, b, c, d):
    h_list = []
    while d > 1:
        d -= 1
        h_list.append(h(a, b, c, d) / d)
    return array(h_list)


@jit(nopython=True)
def g(a, b, c, d):
    return g0(a, b, c) + sum(hiter(a, b, c, d))


def prob_bb_greater_exact(alpha_1, beta_1, m_1, n_1,
                          alpha_2, beta_2, m_2, n_2):

    return g(a=alpha_1 + m_1, b=beta_1 + n_1 - m_1,
             c=alpha_2 + m_2, d=beta_2 + n_2 - m_2)
