from typing import TypeVar, Optional

from pandas import Series

from probability.distributions.protocols import SupportsDataDistribution

DSM = TypeVar('DSM', bound=SupportsDataDistribution)


class DataSortableMixin(object):

    _data: Series

    def upper(
            self: DSM,
            n: Optional[int] = None,
            frac: Optional[float] = None
    ) -> DSM:
        """
        Return a distribution with the top n or top frac proportion of values.
        """
        if (
                (n is None and frac is None) or
                (n is not None and frac is not None)
        ):
            raise ValueError('Must pass one of {n, frac}')
        data = self._data.sort_values(ascending=False)
        if n is not None:
            if not 0 <= n <= len(self):
                raise ValueError(f'n must be between 0 and {len(self)}')
            return type(self)(data=data.head(n))
        else:
            if not 0.0 <= frac <= 1.0:
                raise ValueError('frac must be between 0.0 and 1.0')
            n = int(len(self) * frac)
            return type(self)(data=data.head(n))

    def lower(
            self: DSM,
            n: Optional[int] = None,
            frac: Optional[float] = None
    ) -> DSM:
        """
        Return a distribution with the bottom n or bottom frac proportion of
        values.
        """
        if (
                (n is None and frac is None) or
                (n is not None and frac is not None)
        ):
            raise ValueError('Must pass one of {n, frac}')
        data = self._data.sort_values(ascending=True)
        if n is not None:
            if not 0 <= n <= len(self):
                raise ValueError(f'n must be between 0 and {len(self)}')
            return type(self)(data=data.head(n))
        else:
            if not 0.0 <= frac <= 1.0:
                raise ValueError('frac must be between 0.0 and 1.0')
            n = int(len(self) * frac)
            return type(self)(data=data.head(n))

