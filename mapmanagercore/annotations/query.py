from ..benchmark import timer
from ..loader.base import setColumnTypes
from ..config import Segment
from ..layers.utils import offsetCurveZ
from .utils.queryable import QueryableInterface, queryable
import pandas as pd
from ..utils import polygonUnion
from ..layers.line import calcSubLine, extend
from .mutation import AnnotationsBaseMut
from shapely.geometry import LineString, MultiPolygon
import shapely
import geopandas as gp


class QueryAnnotations(AnnotationsBaseMut, QueryableInterface):
    @property
    def segments(self):
        return SegmentQuery(self[self["segmentID"].drop_duplicates().index])

    @queryable(title="Spine ID", categorical=True)
    def spineID(self):
        return pd.Series(self._points.index.get_level_values(0), index=self._points.index, name="spineID")

    @queryable(title="Time")
    def t(self):
        return pd.Series(self._points.index.get_level_values(1), index=self._points.index, name="time")

    @queryable(title="Segment ID", categorical=True)
    def segmentID(self):
        return self._points["segmentID"]

    @queryable(title="x")
    def x(self):
        return gp.GeoSeries(self._points["point"]).x

    @queryable(title="y")
    def y(self):
        return gp.GeoSeries(self._points["point"]).y

    @queryable(title="z")
    def z(self):
        return self._points["z"]

    @queryable(title="Note", plot=False)
    def note(self):
        return self._points["note"]

    @queryable(title="User Type", categorical=True)
    def userType(self):
        return self._points["userType"]

    @queryable(title="Anchor X")
    def anchorX(self):
        return gp.GeoSeries(self._points["anchor"]).x

    @queryable(title="Anchor Y")
    def anchorY(self):
        return gp.GeoSeries(self._points["anchor"]).y

    @queryable(title="Anchor Z")
    def anchorZ(self):
        return self._points["anchorZ"]

    @queryable(title="Accept", categorical=True, colors={
        True: [255, 0, 0],
        False: [255, 255, 255]
    }, symbols={
        True: "circle",
        False: "cross"
    })
    def accept(self):
        return self._points["accept"]

    @queryable(title="Spine Length")
    def spineLength(self):
        return self._points.apply(lambda x: round(LineString(
            [x["anchor"], x["point"]]).length, 2), axis=1)

    @queryable(title="X Background Offset")
    def xBackgroundOffset(self):
        return self._points["xBackgroundOffset"]

    @queryable(title="Y Background Offset")
    def yBackgroundOffset(self):
        return self._points["yBackgroundOffset"]

    @queryable(title="ROI Head Extend")
    def roiExtend(self):
        return self._points["roiExtend"]

    @queryable(title="Point", plot=False)
    def points(self):
        return self._points["point"]

    @queryable(title="Anchor", plot=False)
    def anchors(self):
        return self._points.apply(lambda x: LineString([x["anchor"], x["point"]]), axis=1)

    @queryable(title="Anchor Point", plot=False)
    def anchorPoint(self):
        return self._points["anchor"]

    def _segments(self):
        invalid = self._lineSegments.index if not ".dep.m" in self._lineSegments.columns else (self._lineSegments.index[self._lineSegments[
            ".dep.m"] != self._lineSegments["modified"]])

        if not invalid.empty:
            self._lineSegments.loc[invalid, "segmentLeft"] = self._lineSegments.loc[invalid].apply(
                lambda x: offsetCurveZ(x["segment"], x["radius"]), axis=1)
            self._lineSegments.loc[invalid, "segmentRight"] = self._lineSegments.loc[invalid].apply(
                lambda x: offsetCurveZ(x["segment"], -x["radius"]), axis=1)
            self._lineSegments.loc[invalid,
                                   ".dep.m"] = self._lineSegments.loc[invalid, "modified"]

        segments = self._points[["segmentID"]].apply(
            lambda d: self._lineSegments.loc[(d["segmentID"], d.name[1])], axis=1)
        return segments if not segments.empty else setColumnTypes(segments, Segment)

    @queryable(title="Segment", plot=False)
    def segment(self):
        return self._segments()["segment"]

    @queryable(title="Left Segment", plot=False)
    def segmentLeft(self):
        return self._segments()["segmentLeft"]

    @queryable(title="Right Segment", plot=False)
    def segmentRight(self):
        return self._segments()["segmentRight"]

    @queryable(title="Radius", segmentDependencies=["radius"])
    def radius(self):
        return self._segments()["radius"]

    @queryable(dependencies=["anchor", "radius"], segmentDependencies=["segment"], plot=False)
    def roiBase(self) -> gp.GeoSeries:
        df = self._points.copy()
        df["segment"] = self._segments()["segment"]
        return df.apply(lambda d: calcSubLine(d["segment"], d["anchor"], distance=8).buffer(d["radius"], cap_style=2), axis=1)

    @queryable(dependencies=["roiBase", "xBackgroundOffset", "yBackgroundOffset"], plot=False)
    def roiBaseBg(self) -> gp.GeoSeries:
        return self._points.apply(
            lambda x: shapely.affinity.translate(
                x["roiBase"], x["xBackgroundOffset"], x["yBackgroundOffset"]),
            axis=1)

    @queryable(dependencies=["point", "anchor", "roiExtend", "radius", "roiBase"], plot=False)
    def roiHead(self) -> gp.GeoSeries:
        def computeRoiHead(x):
            head = extend(LineString([x["anchor"], x["point"]]), origin=x["anchor"],
                          distance=x["roiExtend"]).buffer(x["radius"], cap_style=2)
            head = head.difference(x["roiBase"])
            if isinstance(head, MultiPolygon):
                head = next(
                    poly for poly in head.geoms if poly.contains(x["point"]))
            return head

        return self._points.apply(computeRoiHead, axis=1)

    @queryable(dependencies=["roiHead", "xBackgroundOffset", "yBackgroundOffset"], plot=False)
    def roiHeadBg(self) -> gp.GeoSeries:
        return self._points.apply(
            lambda x: shapely.affinity.translate(
                x["roiHead"], x["xBackgroundOffset"], x["yBackgroundOffset"]),
            axis=1)

    @queryable(dependencies=["roiBase", "roiHead"], plot=False)
    def roi(self) -> gp.GeoSeries:
        return self.roiBase().combine(self.roiHead(), polygonUnion)

    @queryable(dependencies=["roiBaseBg", "roiHeadBg"], plot=False)
    def roiBg(self) -> gp.GeoSeries:
        return self.roiBaseBg().combine(self.roiHeadBg(), polygonUnion)

    @queryable(title="Roi", dependencies=["roi"], aggregate=['sum', 'max'])
    def roiStats(self, channel: int = 0):
        return self.getShapePixels(self.roi(), channel)

    @queryable(title="Background Roi", dependencies=["roi"], aggregate=['sum', 'max'])
    def roiStatsBg(self, channel: int = 0):
        return self.getShapePixels(self.roi(), channel)


class SegmentQuery:

    def __init__(self, annotations: QueryAnnotations) -> None:
        self.annotations = annotations

    def __getitem__(self, items):
        df = self.annotations.__getitem__(items)
        df.index = self.annotations._points.loc[df.index]["segmentID"]
        return df

    @property
    def columns(self):
        return [col for col in self.annotations.columns if col.startswith('segment') and not col == 'segmentID']

    def _ipython_key_completions_(self):
        return self.columns
