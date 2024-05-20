from typing import Callable, Self, Tuple, Union
import numpy as np
from ..layers.point import PointLayer
from .layer import Layer
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import substring
import shapely
import geopandas as gp
from ..benchmark import timer


class MultiLineLayer(Layer):
    @Layer.setProperty
    def offset(self, offset: Union[int, Callable[[int], int]]) -> Self:
        ("implemented by decorator", offset)
        return self

    @Layer.setProperty
    def outline(self, outline: Union[int, Callable[[int], int]]) -> Self:
        ("implemented by decorator", outline)
        return self

    @timer
    def normalize(self) -> Self:
        if "offset" in self.properties:
            distance = self.properties["offset"]
            distance = self.series.index.map(
                lambda x: distance(x))if callable(distance) else distance
            self.series = shapely.offset_curve(self.series, distance=distance)

        if "outline" in self.properties:
            distance = self.properties["outline"]
            distance = self.series.index.map(
                lambda x: distance(x) if callable(distance) else distance)
            self.series = gp.GeoSeries(self.series).buffer(
                distance=distance, cap_style='flat')

        return super().normalize()

    def _encodeBin(self):
        featureId = self.series.index
        coords = self.series
        coords = coords.reset_index(drop=True)
        pathIndices = coords.count_coordinates().cumsum()
        coords = coords.get_coordinates()
        return {"lines": {
            "ids": featureId,
            "featureIds": coords.index.to_numpy(dtype=np.uint16),
            "pathIndices": np.insert(pathIndices.to_numpy(dtype=np.uint16), 0, 0, axis=0),
            "positions": coords.to_numpy(dtype=np.float32).flatten(),
        }}


class LineLayer(MultiLineLayer):
    # clip the shapes z axis
    def clipZ(self, zRange: Tuple[int, int]) -> MultiLineLayer:
        self.series = self.series.apply(clipLine, zRange=zRange)
        self.series.dropna(inplace=True)
        return MultiLineLayer(self)

    @ timer
    def createSubLine(df: gp.GeoDataFrame, distance: int, linc: str, originc: str) -> Self:
        series = df.apply(lambda d: calcSubLine(
            d[linc], d[originc], distance), axis=1)
        return LineLayer(series)

    @ timer
    def subLine(self, distance: int) -> Self:
        self.series = self.series.apply(lambda d: calcSubLine(
            d, getTail(d), distance))
        return self

    @ timer
    def simplify(self, res: int) -> Self:
        self.series = self.series.simplify(res)
        return self

    def extend(self, distance=0.5, originIdx=0) -> Self:
        if isinstance(distance, gp.GeoSeries):
            self.series = self.series.combine(distance, lambda x, distance: extend(
                x, x.coords[originIdx], distance=distance))
        else:
            self.series = self.series.apply(
                lambda x: extend(x, x.coords[originIdx], distance=distance))
        return self

    def tail(self):
        points = PointLayer(self)
        points.series = points.series.apply(lambda x: Point(x.coords[-1]))
        return points

    def head(self):
        points = PointLayer(self)
        points.series = points.series.apply(lambda x: Point(x.coords[0]))
        return points

@timer
def getTail(d):
    return Point(d.coords[1][0], d.coords[1][1])

# abj
def getSide(a: Point, b: Point, c: Point):
  """ Calculate which side a point (c) is relative to a segment (AB)
  Args:
    a: Beginning point of line segment
    b: End point of line segment
    c: Point relative to line segment
  """
  crossProduct = (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)
  if crossProduct > 0:
    return "Right"
  elif crossProduct < 0:
    return "Left"
  else:
    return "On the Line"

# abj
@ timer
def getSpineSide(line: LineString, spine: Point):
    """ Return a string representing the side at which the spine point is relative to its segment

    Args:
        Line: segment in the for of a LineString
        Spine: point
    """
    first = Point(line.coords[0])
    last = Point(line.coords[-1])
    val = getSide(first, last, spine)
    return val

# abb
@ timer
def getSpinePositon(line: LineString, origin: Point):
    """Get the position of a spine anchor on the segment.
    """
    root = line.project(origin)
    return root

@ timer
def calcSubLine(line: LineLayer, origin: Point, distance: int):
    root = line.project(origin)
    sub = substring(line, start_dist=max(
        root - distance, 0), end_dist=root + distance)
    return sub

@timer
def extend(x: LineString, origin: Point, distance: float) -> Polygon:
    scale = 1 + distance / x.length
    # grow by scaler from one direction
    return shapely.affinity.scale(x, xfact=scale, yfact=scale, origin=origin)

@timer
def pushLine(segment, lines):
    if len(segment) <= 1:
        return
    lines.append(segment)


@timer
def clipLine(line: LineString, zRange: Tuple[int, int]):
    z_min, z_max = zRange

    zInRange = [z_min <= p[2] < z_max for p in line.coords]
    if not any(zInRange):
        return None

    # Initialize a list to store the clipped 2D LineString segments
    lines = []
    segment = []

    # Iterate through the line coordinates
    for i in range(len(line.coords) - 1):
        z1InRange, z2InRange = zInRange[i], zInRange[i+1]
        p1 = line.coords[i]

        # Check if the segment is within the z-coordinate bounds
        if z1InRange:
            # Include the entire segment in the clipped 2D LineString
            segment.append((p1[0], p1[1]))

            if not z2InRange:
                # The segment exits the bounds
                point = interpolateAcross(z_min, z_max, p1, line.coords[i+1])
                segment.append(point)

            continue

        p2 = line.coords[i+1]
        if z2InRange:
            # The segment enters the bounds
            point = interpolateAcross(z_min, z_max, p2, p1)
            segment.append(point)
        elif (p1[2] < z_min and p2[2] > z_max) or (p2[2] < z_min and p1[2] > z_max):
            # The segment crosses the z bounds; clip and include both parts
            segment.extend((interpolate(p1, p2, z_min),
                           interpolate(p1, p2, z_max)))

        if len(segment) != 0:
            pushLine(segment, lines)
            segment = []

    if zInRange[-1]:
        x, y, z = line.coords[-1]
        segment.append((x, y))

    pushLine(segment, lines)

    if not lines:
        return None

    if len(lines) == 1:
        return LineString(lines[0])

    return MultiLineString(lines)


# 1 is in and 2 is out
@timer
def interpolateAcross(z_min, z_max, p1, p2):
    if p2[2] >= z_max:
        return interpolate(p1, p2, z_max)
    return interpolate(p1, p2, z_min)


@timer
def interpolate(p1, p2, crossZ):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    t = (crossZ - z1) / (z2 - z1)

    x_interpolated = x1 + t * (x2 - x1)
    y_interpolated = y1 + t * (y2 - y1)
    return (x_interpolated, y_interpolated)
