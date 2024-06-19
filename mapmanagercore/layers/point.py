from typing import Callable, Self, Tuple, Union
import numpy as np
import geopandas as gp
from shapely.geometry import LineString
from mapmanagercore.benchmark import timer
from mapmanagercore.utils import force_2d
from .layer import Layer
from .utils import inRange


class PointLayer(Layer):
    # clip the shapes z axis
    @timer
    def clipZ(self, range: Tuple[int, int]) -> Self:
        self.series = self.series[inRange(self.series.z, range=range)]
        self.series = force_2d(self.series)
        return self

    def splitZ(self, range: Tuple[int, int]) -> Tuple[Self, Self]:
        mask = inRange(self.series.z, range=range)
        self.series = force_2d(self.series)
        return (self.copy(self.series[mask]), self.setSeries(self.series[~mask]))

    @timer
    def toLine(self, points: gp.GeoSeries):
        from .line import LineLayer
        self.series = points.combine(
            self.series, lambda x, x1: LineString([x, x1]))
        return LineLayer(self)

    @Layer.setProperty
    def radius(self, radius: Union[int, Callable[[int], int]]) -> Self:
        ("implemented by decorator", radius)
        return self

    """Adds text labels using the index of the series
    """
    @Layer.setProperty
    def label(self, show=True) -> Self:
        ("implemented by decorator", show)
        return self

    @timer
    def _encodeBin(self):
        coords = self.series.get_coordinates()
        featureId = coords.index
        coords = coords.reset_index(drop=True)
        return {"points": {
            "ids": featureId,
            "featureIds": coords.index.to_numpy(dtype=np.uint16),
            "positions": coords.to_numpy(dtype=np.float32).flatten(),
        }}
