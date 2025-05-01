import dataclasses
from typing import Tuple, Union
from shapely.geometry import LineString, Point
import geopandas as gpd
import numpy as np

from mapmanagercore.utils import interpolate
from mapmanagercore.layers.line import calculateSegmentOffset
from ..lazy_geo_pandas import schema, field, compute, LazyGeoFrame, Schema
from mapmanagercore.logger import logger

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
    
    color: str = field(
        # abb 202504
        default="#FF0000",
        type="str",
        title="Segment Color",
        description="Color of the segment",
        plot=False
    )

    # color: Tuple[int, int, int, int] = field(
    #     # abb 202504
    #     # default=(255, 0, 0),
    #     default=(255, 0, 0, 0),
    #     type="Tuple[int, int, int, int]",
    #     title="Segment Color",
    #     description="Color of the segment",
    #     plot=False
    # )

    @compute(title="Points", dependencies=["segment", "roughTracing"])
    def points(frame: LazyGeoFrame):
        """ Returns the amount of points within the segment
        """
        # Create a column that checks if Segment is only one point
        df = frame[["segment", "roughTracing"]]
        df["isPoint"] = df["roughTracing"].apply(lambda roughT: isinstance(roughT, Point)).astype(int)

        # only count segment when it is not just a singular point get 
        df["pointCount"] = df[["isPoint", "segment"]].apply( lambda row: len(row["segment"].coords) if not row["isPoint"] else 1,
            axis=1
        )
        return df["pointCount"]
    
    @compute(title="Length", dependencies=["segment", "points"])
    def length(frame: LazyGeoFrame):
        """ Returns the length of the segment
        """
        return frame['segment'].apply(
            lambda segment: round(segment.length, 2) if segment.length > 0 else 0
        )

    @compute(title="Pivot Point", dependencies=["segment", "pivotDistance"])
    def pivotPoint(frame: LazyGeoFrame):
        return interpolate(frame['segment'], frame['pivotDistance'])

    # abj
    @compute(title="Left Radius", dependencies=["segment", "radius"])
    def leftRadius(frame: LazyGeoFrame) -> gpd.GeoSeries:
        """
        Returns:
            Geoseries of ['x', 'y', 'z'] which is the segment after offset from centerline.
        """
        # abb TODO merge leftRadius and rightRadius (just pass in switch to do one or the other)
        #  they are syymetric left/right

        df = frame[["segment", "radius"]]
        df["z"] = (df['segment'].apply(lambda geom: [coord[2] for coord in geom.coords] if geom is not None else []))  
        offsettedSegment = df.apply(lambda d: calculateSegmentOffset(d["segment"], d["radius"], isPositive=False), axis=1)
        # logger.info(f"offsettedSegment is: {offsettedSegment}")
        df["x"] = (offsettedSegment.apply(lambda geom: [coord[0] for coord in geom.coords] if geom is not None else []))
        df["y"] = (offsettedSegment.apply(lambda geom: [coord[1] for coord in geom.coords] if geom is not None else []))
        newDF = gpd.GeoSeries(df[["x", "y", "z"]].apply(lambda ldf: LineString(Point(ldf["x"][i], ldf["y"][i], ldf["z"][i])
                                                                            for i, val in enumerate(ldf["x"])), axis=1))
        return newDF
    
    @compute(title="Right Radius", dependencies=["segment", "radius"])
    def rightRadius(frame: LazyGeoFrame):
        # abb TODO merge leftRadius and rightRadius (just pass in switch to do one or the other)
        #  they are syymetric left/right

        df = frame[["segment", "radius"]]
        # logger.info(f" df[radius] {df['radius']}")
        df["z"] = (df['segment'].apply(lambda geom: [coord[2] for coord in geom.coords] if geom is not None else []))  
        offsettedSegment = df.apply(lambda d: calculateSegmentOffset(d["segment"], d["radius"], isPositive=True), axis=1)
        df["x"] = (offsettedSegment.apply(lambda geom: [coord[0] for coord in geom.coords] if geom is not None else []))
        df["y"] = (offsettedSegment.apply(lambda geom: [coord[1] for coord in geom.coords] if geom is not None else []))

        newDF = gpd.GeoSeries(df[["x", "y", "z"]].apply(lambda ldf: LineString(Point(ldf["x"][i], ldf["y"][i], ldf["z"][i]) 
                                                                            for i, val in enumerate(ldf["x"])), axis=1))
        return newDF
    
    @compute(title="Segment Length", dependencies=["segment", "segmentID"])
    def length(frame: LazyGeoFrame) -> float: 
        """ Calculate the length of the segment for plotting segment in dendrogram widget

        Return:
            len of segment in float form
        """
        segment = frame['segment'] 
        return segment.length
        
    # abb do we need this?
    # @compute(title="distance", dependencies=["segment"])
    # def distance(frame: LazyGeoFrame): # distance of each point from beginning of the segment
    #     df = frame["segment"]
    #     distanceList = df.apply(lambda d: getRunningDistance(d))
    #     # distanceList = df.apply(lambda d: getRunningDistance(d["segment"]))
    #     # list of distances, same length as segment: linestring
    #     return distanceList



