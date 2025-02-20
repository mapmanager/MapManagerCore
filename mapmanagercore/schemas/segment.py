import dataclasses
from typing import Union
from shapely.geometry import LineString, Point
from mapmanagercore.utils import interpolate
import numpy as np
from mapmanagercore.layers.line import calculateSegmentOffset
from ..lazy_geo_pandas import schema, field, compute, LazyGeoFrame, Schema

@schema(
    index=["segmentID", "t"],
)
@dataclasses.dataclass
class Segment(Schema):
    """A schema representing a segment"""

    segmentID: int = field(
        title="Segment ID",
        description="Unique identifier for each segment",
        categorical=True
    )
    t: int = field(
        title="Time",
        description="Time of the segment"
    )

    segment: LineString = field(
        title="Segment",
        description="Segment of the spine",
        plot=False
    )
    roughTracing: Union[LineString, Point] = field(
        title="Rough Tracing",
        description="Rough tracing of the spine",
        plot=False
    )

    radius: float = field(
        title="Radius",
        description="Radius of the segment (points)"
    )
    modified: np.datetime64 = field(
        title="Modified",
        description="Time of last modification",
        plot=False
    )

    pivotDistance: float = field(
        default=0.0,
        title="Pivot Distance",
        description="Distance from the pivot point"
    )

    @compute(title="Pivot Point", dependencies=["segment", "pivotDistance"])
    def pivotPoint(frame: LazyGeoFrame):
        return interpolate(frame['segment'], frame['pivotDistance'])

    # abj
    @compute(title="Left Radius", dependencies=["segment", "radius"])
    def leftRadius(frame: LazyGeoFrame):
        df = frame[["segment", "radius"]]
        return calculateSegmentOffset(df["segment"], df["radius"], isPositive=False)

    @compute(title="Right Radius", dependencies=["segment", "radius"])
    def rightRadius(frame: LazyGeoFrame):
        df = frame[["segment", "radius"]]
        return calculateSegmentOffset(df["segment"], df["radius"], isPositive=True)