
import dataclasses
from mapmanagercore.lazy_geo_pandas import schema, Schema
from mapmanagercore.lazy_geo_pandas.schema import field

@schema()
@dataclasses.dataclass
class AnalysisParameters(Schema):
    brightestPathDistance: int = field(
        default=10,
        title="Brightest Path Distance",
        description='points along the tracing to find spine connection (anchor).'
    )

    channel: int = field(
        default=1,
        title="Channel",
        description='image color channel to find brightest connection of spine.'
    )

    zSpread: int = field(
        default=3,
        title="Z Spread",
        description='Number of image slices for max project to find brightest connection of spine.'
    )

    roiExtend: float = field(
        default=4,
        title="ROI Extend",
        description='Number of pixels to extend spine head for spine ROI.'
    )

    roiRadius: float = field(
        default=4,
        title="ROI Radius",
        description='Width of spine ROI.'
    )

    segmentRadius: float = field(
        default=2,
        title="Segment Radius",
        description='Radius of segment tracing.'
    )

    segmentTracingMaxDistance: int = field(
        default=90,
        title="Segment Tracing Max Distance",
        description='Max distance to trace a brightest path with relatively low performance cost.'
    )

    maxChannels: int = field(
        default=2,
        title="Max Channels",
        description='Max number of channels.'
    )

    backgroundRoiGridPoints: int = field(
        default=5,
        title="Background ROI Grid Points",
        description='Number of points used when calculating background ROI. Number of points (n), where grid is n x n'
    )

    backgroundRoiGridOverlap: float = field(
        default=0.1,
        title="Background ROI Grid Overlap",
        description='Value that the background grid points are allowed to overlap'
    )

    # abb for backward compatibility with original class AnalysisParams()
    # todo: remove
    __version__: float = field(
        default=0.0,
        title="xxx",
        description='xxx'
    )