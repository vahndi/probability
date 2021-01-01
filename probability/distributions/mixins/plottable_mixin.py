from typing import List


class PlottableMixin(object):

    x_label: str
    _label: str = None
    y_label: str

    @property
    def label(self):
        if self._label is not None:
            return self._label
        else:
            return str(self)

    def with_x_label(self, x_label: str) -> 'PlottableMixin':

        self.x_label = x_label
        return self

    def with_y_label(self, y_label: str) -> 'PlottableMixin':

        self.y_label = y_label
        return self

    def prepend_to_label(self, prepend: str):

        label = str(self) if self._label is None else self._label
        label = prepend + label
        self._label = label
        return self

    def with_label(self, label: str) -> 'PlottableMixin':

        self._label = label
        return self


class DiscretePlottableMixin(PlottableMixin):

    x_label: str = 'k'
    y_label: str = '$P(K = k)$'


class ContinuousPlottableMixin(PlottableMixin):

    x_label: str = 'x'
    y_label: str = '$P(X = x)$'


class ContinuousPlottableNdMixin(PlottableMixin):

    names: List[str]
    x_label: str = 'x'
    y_label: str = '$P(X = x)$'

    def __getitem__(self, item) -> ContinuousPlottableMixin:

        raise NotImplementedError
