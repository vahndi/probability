"""
Naming Convention: [Big]ST[D]Mixin

Big = Is a capital letter
S = Parameter name
T = Data Type
D = Has a distribution to reset
"""

from typing import Callable


class AFloatDMixin(object):

    _a: float
    _reset_distribution: Callable

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, value: float):
        self._a = value
        self._reset_distribution()


class AlphaFloatMixin(object):

    _alpha: float

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value


class AlphaFloatDMixin(object):

    _alpha: float
    _reset_distribution: Callable

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self._reset_distribution()


class BFloatDMixin(object):

    _b: float
    _reset_distribution: Callable

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, value: float):
        self._b = value
        self._reset_distribution()


class BetaFloatMixin(object):

    _beta: float

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value


class BetaFloatDMixin(object):

    _beta: float
    _reset_distribution: Callable

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        self._reset_distribution()


class CFloatDMixin(object):

    _c: float
    _reset_distribution: Callable

    @property
    def c(self) -> float:
        return self._c

    @c.setter
    def c(self, value: float):
        self._c = value
        self._reset_distribution()


class KIntMixin(object):

    _k: int

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int):
        self._k = value


class BigKIntDMixin(object):

    _K: int
    _reset_distribution: Callable

    @property
    def K(self) -> float:
        return self._K

    @K.setter
    def K(self, value: int):
        self._K = value
        self._reset_distribution()


class LambdaFloatDMixin(object):

    _lambda: float
    _reset_distribution: Callable

    @property
    def lambda_(self) -> float:
        return self._lambda

    @lambda_.setter
    def lambda_(self, value: float):
        self._lambda = value
        self._reset_distribution()


class MuFloatDMixin(object):

    _mu: float
    _reset_distribution: Callable

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        self._mu = value
        self._reset_distribution()


class BigMIntDMixin(object):

    _M: int
    _reset_distribution: Callable

    @property
    def M(self) -> float:
        return self._M

    @M.setter
    def M(self, value: int):
        self._M = value
        self._reset_distribution()


class NIntMixin(object):

    _n: int

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value


class NIntDMixin(object):

    _n: int
    _reset_distribution: Callable

    @property
    def n(self) -> float:
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value
        self._reset_distribution()


class BigNIntDMixin(object):

    _N: int
    _reset_distribution: Callable

    @property
    def N(self) -> float:
        return self._N

    @N.setter
    def N(self, value: int):
        self._N = value
        self._reset_distribution()


class PFloatDMixin(object):

    _p: float
    _reset_distribution: Callable

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, value: float):
        self._p = value
        self._reset_distribution()


class SigmaFloatDMixin(object):

    _sigma: float
    _reset_distribution: Callable

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value: float):
        self._sigma = value
        self._reset_distribution()
