from typing import Iterable, List, Union

from matplotlib.colors import to_rgba

from probability.plotting.types import Color


def cross_fade(
        from_color: Color,
        to_color: Color,
        amount: Union[float, Iterable[float]],
) -> Union[Color, List[Color]]:
    """
    Return a new color (or list of colors) which fades `amount` proportion of
    the way between `from_color` and `to_color`.

    :param from_color: The color to fade from.
    :param to_color: The color to fade to.
    :param amount: The amount to fade by, from 0.0 to 1.0 (scalar or iterable).
    """
    if isinstance(amount, Iterable) and not isinstance(amount, (str, bytes)):
        return [
            cross_fade(from_color, to_color, amt)
            for amt in amount
        ]
    rgba_from = to_rgba(from_color)
    rgba_to = to_rgba(to_color)
    amt = float(amount)
    return tuple(
        from_val + amt * (to_val - from_val)
        for from_val, to_val in zip(rgba_from, rgba_to)
    )


def set_alpha(color: Color, alpha: float) -> Color:
    """
    Set the alpha level of a color to a given value.

    :param color: The existing color.
    :param alpha: The new alpha level.
    """
    rgba = to_rgba(color)
    return (rgba[0], rgba[1], rgba[2], float(alpha))
