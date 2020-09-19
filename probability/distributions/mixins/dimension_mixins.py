from typing import List


class NdMixin(object):

    _names: List[str]

    def _set_names(self, names: List[str]):

        self._names = names

    @property
    def names(self) -> List[str]:

        return self._names
