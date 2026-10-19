from math import atan2, cos, pi, sin
from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from numpy import ndarray
from pandas import Series

from probability.plotting.color_utils import cross_fade
from probability.plotting.number_utils import (
    format_as_integer,
)
from probability.plotting.types import Color, FontSize


def new_axes(
        width: Optional[float] = None,
        height: Optional[float] = None,
        constrained_layout: bool = False,
) -> Axes:
    """
    Create and return a new matplotlib Axes instance.

    :param width: Figure width in inches (defaults to 16).
    :param height: Figure height in inches (defaults to 9).
    :param constrained_layout: Whether to use constrained layout.
    """
    width = width or 16
    height = height or 9
    _, ax = plt.subplots(
        figsize=(width, height),
        constrained_layout=constrained_layout,
    )
    return ax


def new_subplots(
        n_rows: int = 1,
        n_cols: int = 1,
        fig_size: Tuple[float, float] = (16, 9),
        share_x: str = 'none',
        share_y: str = 'none',
        constrained_layout: bool = True,
) -> Tuple[Figure, Union[Axes, ndarray]]:
    """
    Create a figure and grid of subplots.

    :param n_rows: Number of subplot rows.
    :param n_cols: Number of subplot columns.
    :param fig_size: Figure (width, height) in inches.
    :param share_x: x-axis sharing mode ('none', 'all', 'row', 'col').
    :param share_y: y-axis sharing mode ('none', 'all', 'row', 'col').
    :param constrained_layout: Whether to use constrained layout.
    """
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=share_x,
        sharey=share_y,
        figsize=fig_size,
        constrained_layout=constrained_layout,
    )
    return fig, axes


def add_rectangle(
        ax: Axes,
        width: float,
        height: float,
        angle: float = 0.0,
        x_left: Optional[float] = None,
        y_bottom: Optional[float] = None,
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        alpha: Optional[float] = None,
        color: Optional[Color] = None,
        edge_color: Optional[Color] = None,
        face_color: Optional[Color] = None,
        fill: bool = True,
        label: Optional[str] = None,
        line_style: Optional[str] = None,
        line_width: Optional[float] = None,
        z_order: Optional[int] = None,
) -> Axes:
    """
    Add a rectangle patch to the given matplotlib Axes.

    Specify either (`x_left`, `y_bottom`) or (`x_center`, `y_center`).
    """
    if (x_left is None) == (x_center is None):
        raise ValueError('Give exactly one of {x_left, x_center}')
    if (y_bottom is None) == (y_center is None):
        raise ValueError('Give exactly one of {y_bottom, y_center}')
    if (x_left is None) != (y_bottom is None):
        raise ValueError(
            'Give either {x_left, y_bottom} or {x_center, y_center}'
        )

    if x_left is None and y_bottom is None:
        x_l = x_center - width / 2
        y_b = y_center - height / 2
        if angle != 0.0:
            r = ((width / 2) ** 2 + (height / 2) ** 2) ** 0.5
            theta = atan2(height, width)
            xc_new = x_l + r * cos(theta + pi * angle / 180)
            yc_new = y_b + r * sin(theta + pi * angle / 180)
            x = x_l + x_center - xc_new
            y = y_b + y_center - yc_new
        else:
            x = x_l
            y = y_b
    else:
        x = x_left
        y = y_bottom

    rect_kwargs = {
        'xy': (x, y),
        'width': width,
        'height': height,
        'angle': angle,
        'fill': fill,
    }
    if alpha is not None:
        rect_kwargs['alpha'] = alpha
    if color is not None:
        rect_kwargs['color'] = color
    if edge_color is not None:
        rect_kwargs['edgecolor'] = edge_color
    if face_color is not None:
        rect_kwargs['facecolor'] = face_color
    if label is not None:
        rect_kwargs['label'] = label
    if line_style is not None:
        rect_kwargs['linestyle'] = line_style
    if line_width is not None:
        rect_kwargs['linewidth'] = line_width
    if z_order is not None:
        rect_kwargs['zorder'] = z_order

    ax.add_patch(Rectangle(**rect_kwargs))
    return ax


def add_v_density(
        ax: Axes,
        x: float,
        y_to_z: Series,
        color: Optional[Color] = 'k',
        color_min: Optional[Color] = None,
        edge_color: Optional[Color] = None,
        width: float = 0.8,
        z_max: Optional[float] = None,
        h_align: str = 'center',
) -> Axes:
    """
    Add a vertical density bar to the plot.

    :param ax: Matplotlib Axes to plot on.
    :param x: The x-coordinate of the bar.
    :param y_to_z: A Series mapping the bar's y-coordinate to its density.
    :param color: The color of the density bar.
    :param color_min: Optional 2nd color to fade out to.
    :param edge_color: Optional color for the outer edge of the density.
    :param width: The bar width.
    :param z_max: Value to scale down densities by to get to [0, 1].
                  Defaults to max value of `y_to_z`.
    :param h_align: Horizontal alignment: one of {'left', 'center', 'right'}.
    """
    if z_max is None:
        z_max = y_to_z.max()
    y = y_to_z.index.to_list()
    y_lowers = y[:-1]
    y_uppers = y[1:]

    if h_align == 'left':
        x_left = x
    elif h_align == 'center':
        x_left = x - width / 2
    elif h_align == 'right':
        x_left = x - width
    else:
        raise ValueError("h_align must be one of {'left', 'center', 'right'}")

    alphas = (
        (y_to_z / z_max).rolling(2).mean().shift(-1).dropna().clip(0.0, 1.0)
    )

    if color_min is None:
        color_min = color
    colors = cross_fade(from_color=color_min, to_color=color, amount=alphas)

    for y_lower, y_upper, alpha, seg_color in zip(
        y_lowers, y_uppers, alphas, colors
    ):
        add_rectangle(
            ax=ax,
            x_left=x_left,
            y_bottom=y_lower,
            width=width,
            height=y_upper - y_lower,
            face_color=seg_color,
            alpha=float(alpha),
            line_width=0,
        )

    if edge_color is not None:
        add_rectangle(
            ax=ax,
            x_left=x_left,
            y_bottom=y_lowers[0],
            width=width,
            height=y_uppers[-1] - y_lowers[0],
            edge_color=edge_color,
            fill=False,
        )

    return ax


def add_h_density(
        ax: Axes,
        y: float,
        x_to_z: Series,
        color: Optional[Color] = 'k',
        color_min: Optional[Color] = None,
        edge_color: Optional[Color] = None,
        height: float = 0.8,
        z_max: Optional[float] = None,
        v_align: str = 'center',
) -> Axes:
    """
    Add a horizontal density bar to the plot.

    :param ax: Matplotlib Axes to plot on.
    :param y: The y-coordinate of the bar.
    :param x_to_z: A Series mapping the bar's x-coordinate to its density.
    :param color: The color of the density bar.
    :param color_min: Optional 2nd color to fade out to.
    :param edge_color: Optional color for the outer edge of the density.
    :param height: The bar height.
    :param z_max: Value to scale down densities by to get to [0, 1].
                  Defaults to max value of `x_to_z`.
    :param v_align: Vertical alignment: one of {'top', 'center', 'bottom'}.
    """
    if z_max is None:
        z_max = x_to_z.max()
    x = x_to_z.index.to_list()
    x_lefts = x[:-1]
    x_rights = x[1:]

    if v_align == 'bottom':
        y_bottom = y
    elif v_align == 'center':
        y_bottom = y - height / 2
    elif v_align == 'top':
        y_bottom = y - height
    else:
        raise ValueError("v_align must be one of {'top', 'center', 'bottom'}")

    alphas = (
        (x_to_z / z_max).rolling(2).mean().shift(-1).dropna().clip(0.0, 1.0)
    )

    if color_min is None:
        color_min = color
    colors = cross_fade(from_color=color_min, to_color=color, amount=alphas)

    for x_left, x_right, alpha, seg_color in zip(
        x_lefts, x_rights, alphas, colors
    ):
        add_rectangle(
            ax=ax,
            x_left=x_left,
            y_bottom=y_bottom,
            width=x_right - x_left,
            height=height,
            face_color=seg_color,
            alpha=float(alpha),
            line_width=0,
        )

    if edge_color is not None:
        add_rectangle(
            ax=ax,
            x_left=x_lefts[0],
            y_bottom=y_bottom,
            width=x_rights[-1] - x_lefts[0],
            height=height,
            edge_color=edge_color,
            fill=False,
        )

    return ax


def add_text(
        ax: Axes,
        x: Union[float, Iterable[float]],
        y: Union[float, Iterable[float]],
        text: Union[str, Iterable[str]],
        h_align: Optional[str] = None,
        v_align: Optional[str] = None,
        font_size: Optional[FontSize] = None,
        color: Optional[Color] = None,
        bbox_edge_color: Optional[Color] = None,
        bbox_face_color: Optional[Color] = None,
        bbox_fill: Optional[bool] = None,
        **kwargs,
) -> Axes:
    """
    Add one or multiple text labels to the Axes.
    """
    is_iter_x = isinstance(x, Iterable) and not isinstance(x, (str, bytes))
    is_iter_y = isinstance(y, Iterable) and not isinstance(y, (str, bytes))
    is_iter_t = (
        isinstance(text, Iterable) and not isinstance(text, (str, bytes))
    )

    if is_iter_x or is_iter_y or is_iter_t:
        xs = list(x) if is_iter_x else [x]
        ys = list(y) if is_iter_y else [y]
        ts = list(text) if is_iter_t else [text]
        n = max(len(xs), len(ys), len(ts))
        if len(xs) == 1:
            xs = xs * n
        if len(ys) == 1:
            ys = ys * n
        if len(ts) == 1:
            ts = ts * n
        for xi, yi, ti in zip(xs, ys, ts):
            add_text(
                ax=ax,
                x=xi,
                y=yi,
                text=ti,
                h_align=h_align,
                v_align=v_align,
                font_size=font_size,
                color=color,
                bbox_edge_color=bbox_edge_color,
                bbox_face_color=bbox_face_color,
                bbox_fill=bbox_fill,
                **kwargs,
            )
        return ax

    text_kwargs = dict(kwargs)
    if h_align is not None:
        text_kwargs['ha'] = h_align
    if v_align is not None:
        text_kwargs['va'] = v_align
    if font_size is not None:
        text_kwargs['fontsize'] = font_size
    if color is not None:
        text_kwargs['color'] = color

    bbox = dict(text_kwargs.pop('bbox', {}))
    if bbox_edge_color is not None:
        bbox['edgecolor'] = bbox_edge_color
    if bbox_face_color is not None:
        bbox['facecolor'] = bbox_face_color
    if bbox_fill is not None:
        bbox['fill'] = bbox_fill
    if bbox:
        text_kwargs['bbox'] = bbox

    ax.text(x, y, text, **text_kwargs)
    return ax


def format_axis_integer(axis: Axis, kmbt: bool = False) -> None:
    """
    Format an Axis so tick labels are displayed as comma-separated integers.
    """
    axis.set_major_formatter(
        FuncFormatter(lambda x, pos: format_as_integer(x, kmbt=kmbt))
    )


def format_axis_percent(
        axis: Axis,
        num_decimals: int = 0,
        multiply_100: bool = True,
) -> None:
    """
    Format an Axis so tick labels are displayed as percentages.
    """
    def _fmt(value, pos):
        val = float(value)
        if multiply_100:
            val *= 100
        return f'{val:.{num_decimals}f}%'

    axis.set_major_formatter(FuncFormatter(_fmt))
