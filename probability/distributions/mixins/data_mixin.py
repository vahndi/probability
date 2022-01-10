from pandas import Series


class DataMixin(
    object
):

    _data: Series

    @property
    def data(self) -> Series:
        """
        Return the underlying data used to construct the Distribution.
        """
        return self._data

    def sample(self, ):