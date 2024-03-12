from typing import Tuple
import geopandas as gp
from ..types import ImageSlice
from ...loader.base import ImageLoader, Loader


class AnnotationsBase:
    images: ImageLoader  # used for the brightest path
    _points: gp.GeoDataFrame
    _lineSegments: gp.GeoDataFrame

    def __init__(self, loader: Loader):
        self._lineSegments = loader.segments()
        self._points = loader.points()
        self.images = loader.images()

    def slices(self, time: int, channel: int, zRange: Tuple[int, int] = None) -> ImageSlice:
        """
        Loads the image data for a slice.

        Args:
          time (int): The time slot index.
          channel (int): The channel index.
          zRange (Tuple[int, int]): The visible z slice range.

        Returns:
          ImageSlice: The image slice.
        """

        if zRange is None:
            zRange = (int(self._points["z"].min()),
                      int(self._points["z"].max()))

        return ImageSlice(self.images.fetchSlices(time, channel, zRange))

    def getPolygonPixels(self, polygons: gp.GeoSeries, channel: int = 0, zExpand: int = 0):
        polygons = polygons.to_frame(name="polygon")
        polygons["z"] = self._points.loc[polygons.index, "z"]
        polygons["t"] = 0  # self._points.loc[polygons.index, "t"]

        return self.images.getPolygons(polygons, channel=channel, zExpand=zExpand)
