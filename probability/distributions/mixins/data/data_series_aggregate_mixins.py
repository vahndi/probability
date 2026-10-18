from pandas import DataFrame, Series


class DataSeriesMinMixin(object):

    _data: Series

    def min(self) -> Series:
        """
        Return a Series of the smallest value in each distribution.
        """
        return Series({
            ix: self._data[ix].min()
            for ix in self._data.index
        })


class DataSeriesMaxMixin(object):

    _data: Series

    def max(self) -> Series:
        """
        Return a Series of the largest value in each distribution.
        """
        return Series({
            ix: self._data[ix].max()
            for ix in self._data.index
        })


class DataSeriesMeanMixin(object):

    _data: Series

    def mean(self) -> Series:
        """
        Return a Series of the mean value in each distribution.
        """
        return Series({
            ix: self._data[ix].mean()
            for ix in self._data.index
        })


class DataSeriesModeMixin(object):

    _data: Series

    def mode(self) -> Series:
        """
        Return a Series of the mode value in each distribution.
        """
        return Series({
            ix: self._data[ix].mode()
            for ix in self._data.index
        })


class DataSeriesMedianMixin(object):

    _data: Series

    def median(self) -> Series:
        """
        Return a Series of the median value in each distribution.
        """
        return Series({
            ix: self._data[ix].median()
            for ix in self._data.index
        })


class DataSeriesStdMixin(object):

    _data: Series

    def std(self) -> Series:
        """
        Return a Series of the standard deviation of each distribution.
        """
        return Series({
            ix: self._data[ix].std()
            for ix in self._data.index
        })


class DataSeriesVarMixin(object):

    _data: Series

    def var(self) -> Series:
        """
        Return a Series of the variance of each distribution.
        """
        return Series({
            ix: self._data[ix].var()
            for ix in self._data.index
        })


class DataSeriesSemMixin(object):

    _data: Series

    def se(self) -> Series:
        """
        Return a Series of the standard error of the mean of each distribution.
        """
        return Series({
            ix: self._data[ix].se()
            for ix in self._data.index
        })

    def sem(self) -> Series:
        """
        Alias for se(): return a Series of the standard error of the mean of
        each distribution.
        """
        return self.se()

    def mean_dist(self, use_t: bool = False) -> Series:
        """
        Return a Series of the sampling distributions of the mean for each
        distribution.
        """
        return Series({
            ix: self._data[ix].mean_dist(use_t=use_t)
            for ix in self._data.index
        })

    def ci(
            self,
            percent: float = 0.95,
            use_t: bool = False
    ) -> DataFrame:
        """
        Return a DataFrame with 'lower' and 'upper' confidence interval bounds
        around the mean of each distribution.
        """
        rows = {}
        for ix in self._data.index:
            low, high = self._data[ix].ci(percent=percent, use_t=use_t)
            rows[ix] = {'lower': low, 'upper': high}
        df = DataFrame.from_dict(rows, orient='index')
        df.index.name = self._data.index.name
        return df

    def cohens_d(
            self,
            control=None,
            pooled: bool = True
    ):
        """
        Calculate standardized mean differences between distributions in the
        Series.

        :param control: Optional key of the control group. If provided, returns
                        a Series of (group - control) standardized differences;
                        otherwise returns a pairwise DataFrame (index=control,
                        columns=treatment).
        :param pooled: If True, weight variances by sample size (n - 1); if
                       False, use unweighted average variance (v1 + v2) / 2.
        """
        keys = list(self._data.index)
        if control is not None:
            ctrl_dist = self._data[control]
            return Series({
                k: self._data[k].cohens_d(ctrl_dist, pooled=pooled)
                for k in keys
            }, name=self._data.name)
        rows = []
        for ctrl in keys:
            for trat in keys:
                rows.append({
                    'control': ctrl,
                    'treatment': trat,
                    'd': self._data[trat].cohens_d(
                        self._data[ctrl], pooled=pooled
                    )
                })
        return DataFrame(rows).pivot(
            index='control', columns='treatment', values='d'
        ).reindex(keys, axis=0).reindex(keys, axis=1)

    def normalized_diff(self, control=None):
        """
        Calculate unweighted standardized mean differences
        (mean_k - mean_ctrl) / sqrt((var_k + var_ctrl) / 2) across the Series.
        """
        return self.cohens_d(control=control, pooled=False)


