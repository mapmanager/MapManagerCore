from typing import Union
from shapely.geometry import LineString, Point
from mapmanagercore.utils import interpolate
import numpy as np
from mapmanagercore.layers.line import calculateSegmentOffset

from ..lazy_geo_pandas import schema, compute, LazyGeoFrame

@schema(
    index=["segmentID", "t"],
    properties={
        "t": {
            "title": "Time",
            "description": "Time of the segment"
        },
        "segmentID": {
            "categorical": True,
            "title": "Segment ID",
            "description": "Unique identifier for each segment"
        },
        "segment": {
            "title": "Segment",
            "description": "Segment of the spine",
            "plot": False
        },
        "roughTracing": {
            "title": "Rough Tracing",
            "description": "Rough tracing of the spine",
            "plot": False
        },
        "radius": {
            "title": "Radius",
            "description": "Radius of the segment (points)"
        },
        "modified": {
            "title": "Modified",
            "description": "Time of last modification",
            "plot": False
        }
    }
)
class Segment:
    """A schema representing a segment"""
    
    segmentID: int
    t: int

    segment: LineString
    roughTracing: Union[LineString, Point]

    radius: float
    modified: np.datetime64

    pivotDistance: float = 0.0 # abj

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
  
    # Unneeded remove
    # @compute(title="distance", dependencies=["segment"])
    # def distance(frame: LazyGeoFrame): # distance of each point from beginning of the segment
    #     df = frame["segment"]
    #     distanceList = df.apply(lambda d: getRunningDistance(d))
    #     # distanceList = df.apply(lambda d: getRunningDistance(d["segment"]))
    #     # list of distances, same length as segment: linestring
    #     return distanceList



