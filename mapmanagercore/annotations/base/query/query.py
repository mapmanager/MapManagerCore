from copy import copy
import numpy as np
import pandas as pd
from ....layers.line import calcSubLine, extend
from .utils import QueryableInterface, queryable
from ..base_mutation import AnnotationsBaseMut
from ...types import AnnotationsOptions
from ....layers.utils import inRange
from ....utils import filterMask
from shapely.geometry import LineString
import shapely
import geopandas as gp


class QueryAnnotations(AnnotationsBaseMut, QueryableInterface):
    def getSegmentsAndSpines(self, options: AnnotationsOptions):
        z_range = options['selection']['z']
        index_filter = options["filters"]
        segments = []

        for (segmentID, points) in self._points.groupby("segmentID"):
            spines = points.index.to_frame(name="id")
            spines["type"] = "Start"
            spines["invisible"] = ~ inRange(points["z"], z_range)
            spines["invisible"] = spines["invisible"] & ~ filterMask(
                points.index, index_filter)

            segments.append({
                "segmentID": segmentID,
                "spines": spines.to_dict('records')
            })

        return segments

    def filter(self, index: pd.Index):
        filtered = copy(self)
        filtered._points = filtered._points[index]
        return filtered

    def __getitem__(self, key):
        return self._points[key]

    def getSpinePosition(self, t: int, spineID: str):
        return list(self._points.loc[spineID, "point"].coords)[0]

    @queryable(title="Spine ID", categorical=True)
    def spineID(self):
        return pd.Series(self._points.index)

    @queryable(title="Segment ID", categorical=True)
    def segmentID(self):
        return self._points["segmentID"]

    @queryable(title="x")
    def pointX(self):
        return gp.GeoSeries(self._points["point"]).x

    @queryable(title="y")
    def pointY(self):
        return gp.GeoSeries(self._points["point"]).y

    @queryable(title="z")
    def pointZ(self):
        return self._points["z"]

    @queryable(title="Anchor X")
    def anchorX(self):
        return gp.GeoSeries(self._points["anchor"]).x

    @queryable(title="Anchor Y")
    def anchorY(self):
        return gp.GeoSeries(self._points["anchor"]).y

    @queryable(title="Anchor Z")
    def anchorY(self):
        return self._points["anchorZ"]

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

    @queryable(title="Point", plotAble=False)
    def points(self):
        return self._points["point"]

    @queryable(title="Anchor", plotAble=False)
    def anchors(self):
        return self._points.apply(lambda x: LineString([x["anchor"], x["point"]]), axis=1)

    @queryable(title="Anchor Point", plotAble=False)
    def anchorPoint(self):
        return self._points["anchor"]

    def _segments(self):
        return self._lineSegments.loc[self._points["segmentID"].drop_duplicates()]

    def segments(self):
        return self._segments()["segment"]

    def segmentsLeft(self):
        return self._segments().apply(lambda x: shapely.offset_curve(x["segment"], x["radius"]), axis=1)

    def segmentsRight(self):
        return self._segments().apply(lambda x: shapely.offset_curve(x["segment"], -x["radius"]), axis=1)

    @queryable(title="Radius", segmentDependencies=["radius"])
    def radius(self):
        return pd.Series(
            self._lineSegments.loc[self._points["segmentID"], "radius"].values,
            index=self._points.index
        )

    @queryable(dependencies=["point", "anchor", "roiExtend", "radius"], plotAble=False)
    def roiHead(self):
        return self._points.apply(
            lambda x: extend(LineString(
                [x["anchor"], x["point"]]), origin=x["anchor"], distance=x["roiExtend"]).buffer(x["radius"], cap_style=2),
            axis=1)

    @queryable(dependencies=["roiHead", "xBackgroundOffset", "yBackgroundOffset"], plotAble=False)
    def roiHeadBg(self):
        return self._points.apply(
            lambda x: shapely.affinity.translate(
                x["roiHead"], x["xBackgroundOffset"], x["yBackgroundOffset"]),
            axis=1)

    @queryable(dependencies=["anchor", "radius"], segmentDependencies=["segment"], plotAble=False)
    def roiBase(self):
        df = self._points.join(self._lineSegments[["segment"]], on="segmentID")
        return df.apply(lambda d: calcSubLine(d["segment"], d["anchor"], distance=8).buffer(d["radius"], cap_style=2), axis=1)

    @queryable(dependencies=["roiBase", "xBackgroundOffset", "yBackgroundOffset"], plotAble=False)
    def roiBaseBg(self):
        return self._points.apply(
            lambda x: shapely.affinity.translate(
                x["roiBase"], x["xBackgroundOffset"], x["yBackgroundOffset"]),
            axis=1)

    @queryable(dependencies=["roiBase", "roiHead"], plotAble=False)
    def roi(self):
        return self.roiBase().union(self.roiHead())

    @queryable(dependencies=["roiBaseBg", "roiHeadBg"], plotAble=False)
    def roiBg(self):
        return self.roiBaseBg().union(self.roiHeadBg())

    def imageStats(self, shapes: gp.GeoSeries, channel: int = 0, zSpread: int = 0):
        images = self.getShapePixels(shapes, channel, zSpread)
        images = images.explode().astype(np.uint64)
        images = images.groupby(level=0)
        return images.aggregate(['sum', 'max'])

    def allChannelImageStats(self, shapes: gp.GeoSeries, zSpread: int = 0):
        channels = self.images.channels()
        return pd.concat([
            self.imageStats(shapes, channel=channel, zSpread=zSpread).add_prefix(f"Channel {channel + 1} ") for channel in range(0, channels)
        ], axis=1)

    @queryable(title="Roi", dependencies=["roi"])
    def _roiStats(self):
        return self.allChannelImageStats(self.roi())

    @queryable(title="Background Roi", dependencies=["roi"])
    def _bgRoiStats(self):
        return self.allChannelImageStats(self.roiBg())
