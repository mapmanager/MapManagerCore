from typing import List, Tuple, Union, Literal, get_args
from plotly.express import colors

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

# SpineId and SegmentId types use as unique identifiers for spines and segments
SpineId = int
SegmentId = int

# Supported symbols
Symbol = Literal[
    'circle',
    'square',
    'diamond',
    'cross',
    'x',
    'pentagon',
    'hexagon',
    'hexagon2',
    'octagon',
    'star',
    'hexagram',
    'hourglass',
    'bowtie',
    'asterisk',
    'hash',
    'arrow'
]

# The list of supported symbols
symbols = get_args(Symbol)


def colorsRGB(colorList: list[str]):
    """Convert a list of color names to RGB tuples.

    Args:
        colorList (list[str]): List of color names

    Returns:
        list[Color]: List of RGB tuples
    """
    colorRGBs, _ = colors.convert_colors_to_same_type(
        colorList, colortype="tuple")
    return scaleColors(colorRGBs, 255)


def scaleColors(colors: list[Color], scale: float) -> list[Color]:
    """Scale a list of colors by a factor.

    Args:
        colors (list[Color]): List of colors to scale
        scale (float): Factor to scale the colors by

    Returns:
        list[Color]: List of scaled colors
    """
    useInt = scale == int(scale)
    if isinstance(colors, tuple):
        return tuple(int(c*scale) if useInt else c * scale for c in colors)
    return [tuple(int(c*scale) if useInt else c * scale for c in color) for color in colors]


class Colors:
    """Default colors for the annotations."""

    selectedSpine: Color = [0, 255, 255]
    spine: Color = [255, 0, 0]
    anchorPoint: Color = [0, 0, 255]
    anchorLine: Color = [0, 0, 255]
    label: Color = [255, 255, 255]
    roiHead: Color = [255, 255, 0]
    roiBase: Color = [255, 100, 0]
    roiHeadBg: Color = [255, 255, 0]
    roiBaseBg: Color = [255, 100, 0]
    backgroundRoiHead: Color = [255, 255, 255]
    backgroundRoiBase: Color = [255, 100, 255]
    segment: Color = [255, 0, 0]
    pendingSegment: Color = [255 * 0.5, 255 * 0.5, 255 * 0.5]
    segmentSelected: Color = [0, 255, 255]
    segmentEditing: Color = [0, 255, 0]
    intractable: Color = [0, 255, 0]
    categorical: List[Color] = colorsRGB(colors.qualitative.Alphabet)
    scalar: List[Color] = colorsRGB(colors.sequential.gray)
    divergent: List[Color] = colorsRGB(colors.diverging.balance)
    transparent: Color = [0, 0, 0, 0]


class Config:
    colors: Colors = Colors
    ghostOpacity: int = 255 * 0.5
    labelExtension: int = 6
    segmentBoldWidth: int = 4
    segmentWidth: int = 2
    segmentLeftRightStrokeWidth: int = 0.5
    roiStrokeWidth: int = 0.5
    pointRadius: int = 2
    pointRadiusEditing: int = 5
    labelOffset: int = 6
