from typing import List, Self, TypedDict, Tuple, Union, Literal, get_args
import numpy as np
from shapely.geometry import LineString, Point
from plotly.express import colors


SpineId = int
SegmentId = int

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

MAX_TRACING_DISTANCE = 30

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

symbols = get_args(Symbol)


class Colors(TypedDict):
    selectedSpine: Color
    spine: Color
    anchorLine: Color
    label: Color
    roiHead: Color
    roiBase: Color
    roiHeadBg: Color
    roiBaseBg: Color
    backgroundRoiHead: Color
    backgroundRoiBase: Color
    segment: Color
    segmentSelected: Color
    segmentEditing: Color
    intractable: Color
    categorical: List[Color]
    scalar: List[Color]
    divergent: List[Color]


TRANSPARENT = [0, 0, 0, 0]


class Config(TypedDict):
    colors: Colors
    ghostOpacity: int
    labelExtension: int
    segmentBoldWidth: int
    segmentWidth: int
    segmentLeftRightStrokeWidth: int
    roiStrokeWidth: int
    pointRadius: int
    pointRadiusEditing: int
    labelOffset: int


def colorsRGB(colorList: list[str]):
    colorRGBs, _ = colors.convert_colors_to_same_type(
        colorList, colortype="tuple")
    return scaleColors(colorRGBs, 255)


def scaleColors(colors: list[Color], scale: float) -> list[Color]:
    useInt = scale == int(scale)
    if isinstance(colors, tuple):
        return tuple(int(c*scale) if useInt else c * scale for c in colors)
    return [tuple(int(c*scale) if useInt else c * scale for c in color) for color in colors]


CONFIG: Config = {
    "colors": {
        "selectedSpine": [0, 255, 255],
        "spine": [255, 0, 0],
        "anchorPoint": [0, 0, 255],
        "anchorLine": [0, 0, 255],
        "label": [255, 255, 255],
        "roiHead": [255, 255, 0],
        "roiBase": [255, 100, 0],
        "roiHeadBg": [255, 255, 0],
        "roiBaseBg": [255, 100, 0],
        "backgroundRoiHead": [255, 255, 255],
        "backgroundRoiBase": [255, 100, 255],
        "segment": [255, 0, 0],
        "pendingSegment": [255 * 0.5, 255 * 0.5, 255 * 0.5],
        "segmentSelected": [0, 255, 255],
        "segmentEditing": [0, 255, 0],
        "intractable": [0, 255, 0],
        "categorical": colorsRGB(colors.qualitative.Alphabet),
        "divergent": colorsRGB(colors.diverging.balance),
        "scalar": colorsRGB(colors.sequential.gray)
    },
    "ghostOpacity": 255 * 0.5,
    "labelExtension": 6,
    "segmentBoldWidth": 4,
    "segmentWidth": 2,
    "segmentLeftRightStrokeWidth": 0.5,
    "roiStrokeWidth": 0.5,
    "pointRadius": 2,
    "pointRadiusEditing": 5,
    "labelOffset": 6
}

COLORS = CONFIG["colors"]


class SizeMetadata(TypedDict):
    x: int
    y: int
    z: int
    c: int


class VoxelMetadata(TypedDict):
    x: float
    y: float
    z: float


class MetadataPhysicalSize(TypedDict):
    x: float
    y: float
    unit: Literal["µm"]


class Metadata(TypedDict):
    size: SizeMetadata
    voxel: VoxelMetadata
    dtype: Literal["Uint16"]
    physicalSize: MetadataPhysicalSize

    # abb not sure I like these abbreviated type
    #   could this be a class with an __init__()
    #   and have the static member variables as proper self.xxx member variables
    # def _prettyPrint(self):
    #     """Convenience to print metadata to terminal.
    #     """
    #     retStr = ''
    #     retStr += f'SizeMetadata: {self.size}' + '\n'
    #     retStr += f'VoxelMetadata: {self.voxel}' + '\n'
    #     retStr += f'dtype: {self.dtype}' + '\n'
    #     retStr += f'MetadataPhysicalSize: {self.physicalSize}' + '\n'
    #     return retStr
