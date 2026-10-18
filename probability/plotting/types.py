from typing import Tuple, Union

Color = Union[
    str,
    Tuple[float, float, float],
    Tuple[float, float, float, float],
]

FontSize = Union[int, float, str]


class FONT_SIZE:
    xx_small: str = 'xx-small'
    x_small: str = 'x-small'
    small: str = 'small'
    medium: str = 'medium'
    large: str = 'large'
    x_large: str = 'x-large'
    xx_large: str = 'xx-large'
