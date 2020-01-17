class PlottableMixin(object):

    x_label: str = 'x'
    _label: str = None

    @property
    def label(self):
        if self._label is not None:
            return self._label
        else:
            return str(self)

    def with_x_label(self, x_label: str) -> 'PlottableMixin':

        self.x_label = x_label
        return self

    def prepend_to_label(self, prepend: str):

        label = str(self) if self._label is None else self._label
        label = prepend + label
        self._label = label
        return self

    def with_label(self, label: str) -> 'PlottableMixin':

        self._label = label
        return self
