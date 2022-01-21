from pandas import Series

from probability.distributions.mixins.data_mixins import DataMixin


class Boolean(DataMixin):

    def __init__(self, data: Series):
        """
        Create a new Boolean distribution.
        """
        self._data: Series = data
