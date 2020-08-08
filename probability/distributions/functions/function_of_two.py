class FunctionOfTwo(object):
    """
    A function of 2 distributions that can be evaluated later at specific
    values.
    """
    def __init__(self, distribution_1, distribution_2, method):

        self._distribution_1 = distribution_1
        self._distribution_2 = distribution_2
        self.method = method

    def at(self, **kwargs):

        return self.method(self._distribution_1, self._distribution_2, **kwargs)
