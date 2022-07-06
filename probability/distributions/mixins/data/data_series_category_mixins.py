from typing import List, Callable

from pandas import Series


class DataSeriesCategoryMixin(object):

    _data: Series

    @property
    def categories(self) -> List[str]:

        return self._data.iloc[0].categories


class DataSeriesCountsMixin(object):

    _data: Series
    keys: Callable[[], List[str]]

    def counts(self) -> Series:

        return Series({
            key: self._data.loc[key].counts()
            for key in self.keys()
        })


class DataSeriesPMFsMixin(object):

    _data: Series
    keys: Callable[[], List[str]]

    def pmfs(self) -> Series:

        return Series({
            key: self._data.loc[key].pmf()
            for key in self.keys()
        })


class DataSeriesPMFBetasMixin(object):

    _data: Series
    keys: Callable[[], List[str]]

    def pmf_betas(self):

        return Series({
            key: self._data.loc[key].pmf_betas()
            for key in self.keys()
        })
