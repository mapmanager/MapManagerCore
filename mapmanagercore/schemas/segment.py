import dataclasses
from typing import Union

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

    # abb 20250318, added back in
    color: str = field(
        default='#00FF00',
        title="Color",
        description="Color for plotting"
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
        # abb was missing
        # abb todo merge leftRadius and rightRadius (just pass in switch to do one or the other)
        #  they are syymetric left/right
        # import geopandas as gpd

        df = frame[["segment", "radius"]]
        df["z"] = (df['segment'].apply(lambda geom: [coord[2] for coord in geom.coords] if geom is not None else []))  
        offsettedSegment = df.apply(lambda d: calculateSegmentOffset(d["segment"], d["radius"], isPositive=False), axis=1)
        logger.info(f"offsettedSegment is: {offsettedSegment}")
        df["x"] = (offsettedSegment.apply(lambda geom: [coord[0] for coord in geom.coords] if geom is not None else []))
        df["y"] = (offsettedSegment.apply(lambda geom: [coord[1] for coord in geom.coords] if geom is not None else []))
        newDF = gpd.GeoSeries(df[["x", "y", "z"]].apply(lambda ldf: LineString(Point(ldf["x"][i], ldf["y"][i], ldf["z"][i])
                                                                            for i, val in enumerate(ldf["x"])), axis=1))
        return newDF
    
    @compute(title="Right Radius", dependencies=["segment", "radius"])
    def rightRadius(frame: LazyGeoFrame):
        logger.warning('abb turned off')
        # print(f'frame: {type(frame)}')  # lazy_geo_pandas.lazy.LazyGeoFrame
        # print(frame)

        # abb was missing
        import geopandas as gpd

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
        logger.info(f" segment.length { segment.length}")
        return segment.length
        

    # abb do we need this?
    # @compute(title="distance", dependencies=["segment"])
    # def distance(frame: LazyGeoFrame): # distance of each point from beginning of the segment
    #     df = frame["segment"]
    #     distanceList = df.apply(lambda d: getRunningDistance(d))
    #     # distanceList = df.apply(lambda d: getRunningDistance(d["segment"]))
    #     # list of distances, same length as segment: linestring
    #     return distanceList



