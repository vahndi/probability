class PlottableMixin(object):

    x_label: str = 'x'

    def with_x_label(self, x_label: str):

        self.x_label = x_label
        return self
