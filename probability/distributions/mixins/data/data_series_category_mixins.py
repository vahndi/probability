from typing import List

from pandas import Series

from probability.distributions.protocols import SupportsDataSeries


class DataSeriesCategoryMixin(object):

    _data: Series

    @property
    def categories(self: SupportsDataSeries) -> List[str]:

        return self._data.iloc[0].categories


class DataSeriesCountsMixin(object):

    def counts(self: SupportsDataSeries) -> Series:

        return Series({
            key: self._data.loc[key].counts()
            for key in self.keys()
        })


class DataSeriesPMFsMixin(object):

    def pmfs(self: SupportsDataSeries) -> Series:

        return Series({
            key: self._data.loc[key].pmf()
            for key in self.keys()
        })


class DataSeriesPMFBetasMixin(object):

    def pmf_betas(self: SupportsDataSeries) -> Series:

        return Series({
            key: self._data.loc[key].pmf_betas()
            for key in self.keys()
        })

