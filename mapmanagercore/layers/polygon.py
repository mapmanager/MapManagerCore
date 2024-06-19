import numpy as np
from mapmanagercore.benchmark import timer
from mapmanagercore.utils import count_coordinates
from .layer import Layer
import geopandas as gp
from shapely.geometry import Polygon, Point

from mapmanagercore.logger import logger

class PolygonLayer(Layer):
    def box(minx, miny, maxx, maxy):
        return PolygonLayer(gp.GeoSeries([Polygon.from_bounds(minx, miny, maxx, maxy)]))

    @timer
    def _encodeBin(self):
        featureId = self.series.index
        coords = self.series
        coords = coords.reset_index(drop=True)
        polygonIndices = count_coordinates(coords).cumsum()
        coords = coords.get_coordinates()

        return {"polygons": {
            "ids": featureId,
            "featureIds": coords.index.to_numpy(dtype=np.uint16),
            "polygonIndices": np.insert(polygonIndices.to_numpy(dtype=np.uint16), 0, 0, axis=0),
            "positions": coords.to_numpy(dtype=np.float32).flatten(),
        }}
