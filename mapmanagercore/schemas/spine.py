from enum import StrEnum
import numpy as np
from mapmanagercore.lazy_geo_pd_images.store import LazyImagesGeoPandas
from mapmanagercore.benchmark import timer
from mapmanagercore.utils import covered_by, union
from ..layers.line import calcSubLine, extend, pointAngle
import shapely
from ..lazy_geo_pandas import schema, compute, LazyGeoFrame
import geopandas as gp
from shapely.geometry import LineString, MultiPolygon, Polygon, Point
from ..lazy_geo_pd_images import aggregateROI

from mapmanagercore.logger import logger

# abb add this to core and don't use 'LEft' 'Right' as string, use this type
# but when we write to a file, use, SpineSide.Left.value -> str
# spioneSideAsStr = SpineSide.Left.value
class SpineSide(StrEnum):
    Left = "Left"
    Right = "Right"
    Undefined = "Undefined"

@schema(
    index=["spineID", "t"],
    relationships={
        "Segment": ["segmentID", "t"]
    },
    properties={
        "spineID": {
            "categorical": True,
            "title": "Spine ID",
            "description": "Unique identifier for each spine",
            "plot": False
        },
        "t": {
            "title": "Time",
            "description": "Time of the spine",
        },
        "segmentID": {
            "categorical": True,
            "title": "Segment ID",
            "description": "Unique identifier for each segment",
            "plot": False
        },
        "point": {
            "title": "Point",
            "description": "Location of the spine",
            "plot": False,
        },
        "anchor": {
            "title": "Anchor",
            "description": "Anchor of the spine",
            "plot": False,
        },
        "xBackgroundOffset": {
            "title": "X Background Offset",
            "description": "X background offset of the spine",
        },
        "yBackgroundOffset": {
            "title": "Y Background Offset",
            "description": "Y background offset of the spine",
        },
        "z": {
            "title": "Z",
            "description": "Z coordinate of the spine",
            "group": "Coordinate"
        },
        "anchorZ": {
            "title": "Anchor Z",
            "description": "Anchor Z coordinate of the spine",
            "group": "Anchor Coordinate",
        },
        "modified": {
            "title": "Modified",
            "description": "Time of last modification",
            "plot": False
        },
        "roiExtend": {
            "title": "ROI Extend",
            "description": "Region of interest extend",
        },
        "roiRadius": {
            "title": "ROI Radius",
            "description": "Region of interest radius",
        },
        "note": {
            "title": "Note",
            "description": "Note about the spine",
            "plot": False
        },
        "userType": {
            "title": "User Type",
            "description": "Type of user",
            "categorical": True,
        },
        "accept": {
            "title": "Accept",
            "description": "Whether the spine is accepted or not",
            "categorical": True,
            "colors": {
                True: [255, 0, 0],
                False: [255, 255, 255]
            },
            "symbols": {
                True: "circle",
                False: "cross"
            }
        },
    }
)
class Spine:
    """A schema representing a spine"""

    spineID: int
    t: int

    segmentID: int
    point: Point
    anchor: Point

    xBackgroundOffset: float
    yBackgroundOffset: float
    z: int
    anchorZ: int
    modified: np.datetime64

    roiExtend: float
    roiRadius: float = 4.0

    # roiRadius: float = dataclasses.field(default=4.0, metadata={
                                                                # "title": "ROI Radius",
                                                                # "description": "Region of interest radius",
    #                                                          })

    note: str = ""
    userType: int = 0
    accept: bool = True

    # Computed columns
    
    @compute(title="X", dependencies=["point"], group="Coordinate")
    @timer
    def x(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["point"]).x

    @compute(title="Y", dependencies=["point"], group="Coordinate")
    @timer
    def y(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["point"]).y

    @compute(title="Anchor X", dependencies=["anchor"], group="Anchor Coordinate")
    @timer
    def anchorX(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["anchor"]).x

    @compute(title="Anchor Y", dependencies=["anchor"], group="Anchor Coordinate")
    @timer
    def anchorY(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["anchor"]).y

    @compute(title="Spine Length", dependencies=["anchor", "point"])
    @timer
    def spineLength(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["anchor"]).distance(frame["point"])

    @compute(title="Spine Position", dependencies={
        "Spine": ["segmentID", "anchor"],
        "Segment": ["segment", "pivotDistance"]
    }, description="Position (distance) of an anchor on the segment", plot=False)
    def spinePosition(frame: LazyGeoFrame):
        # position of spine anchor along the segment
        segmentFrame = frame.getFrame("Segment")

        df = frame[["segmentID", "anchor"]].join(
            segmentFrame[["segment", "pivotDistance"]], on=["segmentID", "t"])

        # Normalize the position of the anchor on the segment
        return shapely.line_locate_point(df["segment"], df["anchor"]) - df["pivotDistance"]

    """abb what is the return type of spineSide function?"""
    # abb use SpineSide enumerated type
    @compute(title="Spine Side", dependencies=
             {
                "Spine": ["segmentID", "point", "anchor"],
                "Segment": ["segment"],
            }, description="Side of spine w.r.t. segment in ('left', 'right')", plot=False)
    def spineSide(frame: LazyGeoFrame):
        segmentFrame = frame.getFrame("Segment")
        df = frame[["segmentID", "anchorLine"]].join(
            segmentFrame[["leftRadius", "rightRadius"]], on=["segmentID", "t"])

        intersectsLeft = shapely.intersects(df["leftRadius"], df["anchorLine"])
        intersectsRight = shapely.intersects(df["rightRadius"], df["anchorLine"])

        # if both are True or both are False, then it is invalid
        valid = intersectsLeft ^ intersectsRight
        _ret = np.where(valid, np.where(intersectsLeft, "Left", "Right"), "Invalid")
        # logger.error(f'_ret:{_ret}')
        return _ret
    
    @compute(title="Anchor", dependencies=["anchor", "point"], plot=False)
    @timer
    def anchorLine(frame: LazyGeoFrame):
        return frame[["anchor", "point"]].apply(lambda x: LineString([x["anchor"], x["point"]]), axis=1)

    @compute(title="Spine Angle", dependencies=["anchor", "point"])
    def spineAngle(frame: LazyGeoFrame):
        return pointAngle(frame["anchor"], frame["point"])

    ## ROI ##

    @compute(tile="ROI Base", dependencies={
        "Spine": ["anchor"],
        "Segment": ["segment", "radius"]
    }, plot=False)
    @timer
    def roiBase(frame: LazyGeoFrame) -> gp.GeoSeries:
        segmentFrame = frame.getFrame("Segment")

        df = frame[["segmentID", "anchor"]].join(
            segmentFrame[["segment", "radius"]], on=["segmentID", "t"])

        return df.apply(lambda d: calcSubLine(d["segment"], d["anchor"], distance=8), axis=1).buffer(df["radius"], cap_style='flat')

    @compute(title="ROI Base Background", dependencies=["roiBase", "xBackgroundOffset", "yBackgroundOffset"], plot=False)
    @timer
    def roiBaseBg(frame: LazyGeoFrame) -> gp.GeoSeries:
        return frame[["roiBase", "xBackgroundOffset", "yBackgroundOffset"]].apply(
            lambda x: shapely.affinity.translate(
                x["roiBase"], x["xBackgroundOffset"], x["yBackgroundOffset"]),
            axis=1)

    @compute(title="ROI Head", dependencies=["point", "anchor", "roiExtend", "roiRadius", "roiBase"], plot=False)
    @timer
    def roiHead(frame: LazyGeoFrame) -> gp.GeoSeries:
        def computeRoiHead(x):
            head = extend(LineString([x["anchor"], x["point"]]), origin=x["anchor"],
                        distance=x["roiExtend"]).buffer(x["roiRadius"], cap_style=2)
            head = head.difference(x["roiBase"])
            if isinstance(head, MultiPolygon):
                for poly in head.geoms:
                    if poly.contains(x["point"]):
                        return poly
                return Polygon()
            return head

        return frame[["point", "anchor", "roiExtend", "roiRadius", "roiBase"]].apply(computeRoiHead, axis=1)

    @compute(title="ROI Head Background", dependencies=["roiHead", "xBackgroundOffset", "yBackgroundOffset"], plot=False)
    @timer
    def roiHeadBg(frame: LazyGeoFrame) -> gp.GeoSeries:
        return frame[["roiHead", "xBackgroundOffset", "yBackgroundOffset"]].apply(
            lambda x: shapely.affinity.translate(
                x["roiHead"], x["xBackgroundOffset"], x["yBackgroundOffset"]),
            axis=1)

    @compute(title="ROI", dependencies=["roiBase", "roiHead"], plot=False)
    @timer
    def roi(frame: LazyGeoFrame) -> gp.GeoSeries:
        return union(frame["roiBase"], frame["roiHead"], grid_size=0.25)

    @compute(title="ROI Background", dependencies=["roiBaseBg", "roiHeadBg"], plot=False)
    @timer
    def roiBg(frame: LazyGeoFrame) -> gp.GeoSeries:
        return union(frame["roiBaseBg"], frame["roiHeadBg"], grid_size=0.25)

    @compute(dependencies=["roi"], plot=False)
    def roiInBounds(frame: LazyGeoFrame) -> gp.GeoSeries:
        imageStore: LazyImagesGeoPandas = frame.getStore()
        _, x, y = imageStore.imageBounds()
        bounds = Polygon([(0, 0), (x, 0), (x, y), (0, y)])
        return covered_by(frame["roi"], bounds)

    @compute(dependencies=["roiBg"], plot=False)
    def roiBgInBounds(frame: LazyGeoFrame) -> gp.GeoSeries:
        imageStore: LazyImagesGeoPandas = frame.getStore()
        _, x, y = imageStore.imageBounds()
        bounds = Polygon([(0, 0), (x, 0), (x, y), (0, y)])
        return covered_by(frame["roiBg"], bounds)
    
    @compute(dependencies=["roiInBounds", "roiBgInBounds"], plot=False)
    def isValid(frame: LazyGeoFrame):
        return frame["roiInBounds"] & frame["roiBgInBounds"]
    
    # abj
    @compute(dependencies=["roiInBounds"], plot=False)
    def intBad(frame: LazyGeoFrame):
        """ Denotes that spine's ROI Intensity is bad. Keep Spine but set its intensity to zero
        """
        return ~(frame["roiInBounds"] .astype(bool))

    ## Image based ROI computed stats ##

    @aggregateROI(title="Roi", dependencies=["roi", "z"], aggregate=['sum', 'max'], group="ROI")
    @timer
    def roiStats(frame: LazyGeoFrame):
        return frame[["roi", "z"]]

    @aggregateROI(title="Background Roi", dependencies=["roiBg", "z"], aggregate=['sum', 'max'], group="ROI Background")
    @timer
    def roiStatsBg(frame: LazyGeoFrame):
        return frame[["roiBg", "z"]]