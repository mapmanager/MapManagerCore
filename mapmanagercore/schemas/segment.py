from typing import Union
from shapely.geometry import LineString, Point
import numpy as np
from ..lazy_geo_pandas import schema


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
        # abb adding left/right radius line to backend
        "leftRadiusLine": {
            "title": "Left Radius",
            "description": "Left radius line",
            "plot": False
        },
        "rightRadiusLine": {
            "title": "Right Radius",
            "description": "Right radius line",
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
    
    # abb swapped order
    segmentID: int
    t: int

    segment: LineString
    roughTracing: Union[LineString, Point]

    # abb
    leftRadiusLine: LineString
    rightRadiusLine: LineString

    radius: float = 4
    modified: np.datetime64
