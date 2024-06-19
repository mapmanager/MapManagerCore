import numpy as np
from shapely.geometry import LineString, Point
from ..benchmark import timer

@timer
def inRange(x, range):
    return np.invert((range[0] > x) | (range[1] < x))

@timer
def roundPoint(point: Point, ndig=0):
    return Point(round(point.x, ndig), round(point.y, ndig), round(point.z, ndig))


@timer
def offsetCurveZ(line: LineString, offset: int) -> LineString:
    offsetLine: LineString = line.parallel_offset(offset, join_style=2)
    points = [(x, y, line.interpolate(line.project(Point(x, y))).z)
              for x, y in offsetLine.coords]
    return LineString(points)
