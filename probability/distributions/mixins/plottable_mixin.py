from typing import List, Optional, TypeVar

SelfT = TypeVar('SelfT', bound='PlottableMixin')


class PlottableMixin(object):

    x_label: str
    _label: Optional[str] = None
    y_label: str

    @property
    def label(self):
        if self._label is not None:
            return self._label
        else:
            return str(self)

    def with_x_label(self: SelfT, x_label: str) -> SelfT:

        self.x_label = x_label
        return self

    def with_y_label(self: SelfT, y_label: str) -> SelfT:

        self.y_label = y_label
        return self

    def prepend_to_label(self: SelfT, prepend: str) -> SelfT:

        label = str(self) if self._label is None else self._label
        label = prepend + label
        self._label = label
        return self

    def with_label(self: SelfT, label: str) -> SelfT:

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
