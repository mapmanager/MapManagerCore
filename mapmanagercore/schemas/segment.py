from typing import Union
from shapely.geometry import LineString, Point
import numpy as np
import geopandas as gpd
from mapmanagercore.logger import logger
from mapmanagercore.layers.line import calculateSegmentOffset

from ..lazy_geo_pandas import schema, compute, LazyGeoFrame
# from ..lazy_geo_pandas import schema


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
        # "leftRadiusLine": {
        #     "title": "Left Radius",
        #     "description": "Left radius line",
        #     "plot": False
        # },
        # "rightRadiusLine": {
        #     "title": "Right Radius",
        #     "description": "Right radius line",
        #     "plot": False
        # },

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

    radius: float
    modified: np.datetime64

    # abj
    @compute(title="Left Radius", dependencies=["segment", "radius"])
    def leftRadius(frame: LazyGeoFrame):
        df = frame[["segment", "radius"]]
        df["z"] = (df['segment'].apply(lambda geom: [coord[2] for coord in geom.coords]))  
        offsettedSegment = df.apply(lambda d: calculateSegmentOffset(d["segment"], d["radius"], isPositive=False), axis=1)
        df["x"] = (offsettedSegment.apply(lambda geom: [coord[0] for coord in geom.coords]))
        df["y"] = (offsettedSegment.apply(lambda geom: [coord[1] for coord in geom.coords]))
        newDF = gpd.GeoSeries(df[["x", "y", "z"]].apply(lambda ldf: LineString(Point(ldf["x"][i], ldf["y"][i], ldf["z"][i]) 
                                                                               for i, val in enumerate(ldf["x"])), axis=1))

        # logger.info(f"newDF {newDF}")
        return newDF
    
    @compute(title="Right Radius", dependencies=["segment", "radius"])
    def rightRadius(frame: LazyGeoFrame):
        df = frame[["segment", "radius"]]
        df["z"] = (df['segment'].apply(lambda geom: [coord[2] for coord in geom.coords]))
        offsettedSegment = df.apply(lambda d: calculateSegmentOffset(d["segment"], d["radius"], isPositive=True), axis=1)
        df["x"] = (offsettedSegment.apply(lambda geom: [coord[0] for coord in geom.coords]))
        df["y"] = (offsettedSegment.apply(lambda geom: [coord[1] for coord in geom.coords]))
        newRightRadiusDF = gpd.GeoSeries(df[["x", "y", "z"]].apply(lambda ldf: LineString(Point(ldf["x"][i], ldf["y"][i], ldf["z"][i]) 
                                                                               for i, val in enumerate(ldf["x"])), axis=1))
        return newRightRadiusDF


