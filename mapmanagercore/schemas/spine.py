from shapely.geometry import Point
import numpy as np
from ..layers.line import calcSubLine, extend
import shapely
from ..lazy_geo_pandas import schema, calculated, LazyGeoFrame
import geopandas as gp
from shapely.geometry import LineString, MultiPolygon
from ..lazy_geo_pd_image import calculatedROI

from mapmanagercore.logger import logger

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
        },
        "t": {
            "title": "Time",
            "description": "Time of the spine",
        },
        "segmentID": {
            "categorical": True,
            "title": "Segment ID",
            "description": "Unique identifier for each segment",
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
        },
        "anchorZ": {
            "title": "Anchor Z",
            "description": "Anchor Z coordinate of the spine",
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

    roiExtend: float = 4.0
    roiRadius: float = 4.0
    note: str = ""
    userType: int = 0
    accept: bool = True

    @calculated(title="x", dependencies=["point"])
    def x(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["point"]).x

    @calculated(title="y", dependencies=["point"])
    def y(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["point"]).y

    @calculated(title="Anchor X", dependencies=["anchor"])
    def anchorX(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["anchor"]).x

    @calculated(title="Anchor Y", dependencies=["anchor"])
    def anchorY(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["anchor"]).y

    @calculated(title="Spine Length", dependencies=["anchor", "point"])
    def spineLength(frame: LazyGeoFrame):
        return gp.GeoSeries(frame["anchor"]).distance(frame["point"])

    @calculated(title="Anchor", dependencies=["anchor", "point"], plot=False)
    def anchorLine(frame: LazyGeoFrame):
        return frame[["anchor", "point"]].apply(lambda x: LineString([x["anchor"], x["point"]]), axis=1)

    @calculated(tile="ROI Base", dependencies={
        "Spine": ["anchor"],
        "Segment": ["segment", "radius"]
    }, plot=False)
    def roiBase(frame: LazyGeoFrame) -> gp.GeoSeries:
        segmentFrame = frame.getFrame("Segment")

        df = frame[["segmentID", "anchor"]].join(
            segmentFrame[["segment", "radius"]], on=["segmentID", "t"])

        return df.apply(lambda d: calcSubLine(d["segment"], d["anchor"], distance=8), axis=1).buffer(df["radius"], cap_style='flat').set_precision(1)

    @calculated(title="ROI Base Background", dependencies=["roiBase", "xBackgroundOffset", "yBackgroundOffset"], plot=False)
    def roiBaseBg(frame: LazyGeoFrame) -> gp.GeoSeries:
        return frame[["roiBase", "xBackgroundOffset", "yBackgroundOffset"]].apply(
            lambda x: shapely.affinity.translate(
                x["roiBase"], x["xBackgroundOffset"], x["yBackgroundOffset"]),
            axis=1)

    @calculated(title="ROI Head", dependencies=["point", "anchor", "roiExtend", "roiRadius", "roiBase"], plot=False)
    def roiHead(frame: LazyGeoFrame) -> gp.GeoSeries:
        def computeRoiHead(x):
            head = extend(LineString([x["anchor"], x["point"]]), origin=x["anchor"],
                          distance=x["roiExtend"]).buffer(x["roiRadius"], cap_style=2)
            head = head.difference(x["roiBase"])
            if isinstance(head, MultiPolygon):
                head = next(
                    poly for poly in head.geoms if poly.contains(x["point"]))
            return head

        return frame[["point", "anchor", "roiExtend", "roiRadius", "roiBase"]].apply(computeRoiHead, axis=1).set_precision(1)

    @calculated(title="ROI Head Background", dependencies=["roiHead", "xBackgroundOffset", "yBackgroundOffset"], plot=False)
    def roiHeadBg(frame: LazyGeoFrame) -> gp.GeoSeries:
        return frame[["roiHead", "xBackgroundOffset", "yBackgroundOffset"]].apply(
            lambda x: shapely.affinity.translate(
                x["roiHead"], x["xBackgroundOffset"], x["yBackgroundOffset"]),
            axis=1)

    @calculated(title="ROI", dependencies=["roiBase", "roiHead"], plot=False)
    def roi(frame: LazyGeoFrame) -> gp.GeoSeries:
        return frame["roiBase"].union(frame["roiHead"])

    @calculated(title="ROI Background", dependencies=["roiBaseBg", "roiHeadBg"], plot=False)
    def roiBg(frame: LazyGeoFrame) -> gp.GeoSeries:
        return frame["roiBaseBg"].union(frame["roiHeadBg"])

    @calculatedROI(title="Roi", dependencies=["roi", "z"], aggregate=['sum', 'max'])
    def roiStats(frame: LazyGeoFrame):
        return frame[["roi", "z"]]

    @calculatedROI(title="Background Roi", dependencies=["roiBg", "z"], aggregate=['sum', 'max'])
    def roiStatsBg(frame: LazyGeoFrame):
        return frame[["roiBg", "z"]]