from typing import Union
import numpy as np
from shapely.geometry import Polygon, LineString, GeometryCollection, MultiPolygon
import shapely
import skimage.draw
import pandas as pd
from .benchmark import timer
import itertools


@timer
def filterMask(d, index_filter):
    if index_filter == None or len(index_filter) == 0:
        return np.full(len(d), False)
    return ~d.isin(index_filter)


@timer
def shapeIndexes(d: Union[Polygon, LineString]):
    d = shapely.force_2d(d)

    # TODO: fully support multi-polygon
    if isinstance(d, MultiPolygon):
        d = d.geoms[0]
    if isinstance(d, GeometryCollection):
        d = next(s for s in d.geoms if isinstance(s, Polygon))

    if isinstance(d, Polygon):
        x, y = zip(*d.exterior.coords)
        return skimage.draw.polygon(x, y)

    x, y = zip(*d.coords)
    return skimage.draw.line(int(x[0]), int(y[0]), int(x[1]), int(y[1]))


def generateGrid(stepX, stepY, points, centerX=0, centerY=0):
    distanceX = stepX * points
    distanceY = stepY * points
    x = np.arange(0, distanceX, stepX) - \
        (stepX * ((distanceX * 0.5) // stepX)) + centerX
    y = np.arange(0, distanceY, stepY) - \
        (stepY * ((distanceY * 0.5) // stepY)) + centerY
    return pd.DataFrame(itertools.product(x, y), columns=["x", "y"])


def shapeGrid(shape, points, overlap=0):
    minx, miny, maxx, maxy = shape.bounds
    width = maxx - minx
    height = maxy - miny
    overlap = 1 - overlap
    return generateGrid(width * overlap, height * overlap, points)
