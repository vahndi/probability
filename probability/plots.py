import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.axes import Axes
from types import FunctionType


def new_axes(width: int = 16, height: int = 9):
    """
    :rtype: Axes
    """
    _, ax = plt.subplots(figsize=(width, height))
    return ax


def set_axis_tick_label_rotation(ax: Axis, rotation: int):
    """
    Set the rotation of axis tick labels.

    :param ax: The axis whose tick label rotation to set.
    :param rotation: The rotation value to set.
    """
    if ax.get_majorticklabels():
        plt.setp(ax.get_majorticklabels(), rotation=rotation)
    if ax.get_minorticklabels():
        plt.setp(ax.get_minorticklabels(), rotation=rotation)


def transform_axis_tick_labels(ax: Axis, transformation: FunctionType):
    """
    Transforms the labels of each label along the axis by a transformation function.

    :param ax: The axis whose tick labels to transform.
    :param transformation: The transformation function e.g. `lambda t: t.split('T')[0]`.
    """
    ax.figure.canvas.draw()  # make sure the figure has been drawn so the labels are available to be got
    labels = ax.get_ticklabels()
    for label in labels:
        new_label = transformation(label.get_text())
        label.set_text(new_label)
    ax.set_ticklabels(labels)
