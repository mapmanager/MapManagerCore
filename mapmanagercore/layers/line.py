from typing import Callable, Self, Tuple, Union
import numpy as np
from mapmanagercore.utils import count_coordinates
from mapmanagercore.logger import logger
from ..layers.point import PointLayer
from .layer import Layer
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import substring
import shapely
import geopandas as gp
from ..benchmark import timer
import math
from math import pi as PI

from mapmanagercore.logger import logger

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

    @timer
    def _encodeBin(self):
        coords = self.series.explode(index_parts=False)
        featureId = coords.index
        coords = coords.reset_index(drop=True)
        pathIndices = count_coordinates(coords).cumsum()
        coords = coords.get_coordinates()
        return {"lines": {
            "ids": featureId,
            "featureIds": coords.index.to_numpy(dtype=np.uint16),
            "pathIndices": np.insert(pathIndices.to_numpy(dtype=np.uint16), 0, 0, axis=0),
            "positions": coords.to_numpy(dtype=np.float32).flatten(),
        }}


class LineLayer(MultiLineLayer):
    @timer
    # clip the shapes z axis
    def clipZ(self, zRange: Tuple[int, int]) -> MultiLineLayer:
        self.series = clipLines(self.series, zRange)
        self.series.dropna(inplace=True)
        return MultiLineLayer(self)

    @timer
    def createSubLine(df: gp.GeoDataFrame, distance: int, linc: str, originc: str) -> Self:
        series = df.apply(lambda d: calcSubLine(
            d[linc], d[originc], distance), axis=1)
        return LineLayer(series)

    @timer
    def subLine(self, distance: int) -> Self:
        self.series = self.series.apply(lambda d: calcSubLine(
            d, getTail(d), distance))
        return self

    @timer
    def simplify(self, res: int) -> Self:
        self.series = self.series.simplify(res)
        return self

    @timer
    def extend(self, distance=0.5, originIdx=0) -> Self:
        if isinstance(distance, gp.GeoSeries):
            self.series = self.series.combine(distance, lambda x, distance: extend(
                x, x.coords[originIdx], distance=distance))
        else:
            self.series = self.series.apply(
                lambda x: extend(x, x.coords[originIdx], distance=distance))
        return self

    @timer
    def tail(self):
        points = PointLayer(self)
        points.series = points.series.apply(lambda x: Point(x.coords[-1]))
        return points

    @timer
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
def getSpineSide(line: LineString, spine: Point, anchor:Point):
    """ Return a string representing the side at which the spine point is relative to its segment

    Args:
        Line: segment in the form of a LineString
        Spine: point
        anchor: point on segment line that spine point connects to
    """
    # logger.info("Calculating spine side in mmc")
    if line is None:
        # logger.error(f'got None line for spine: {spine}')
        return
    
    # old method getting first and last coordinate of segment
    # first = Point(line.coords[0])
    # last = Point(line.coords[-1])

    first = Point(line.coords[0])
    last = anchor

    val = getSide(first, last, spine)
    return val

# abj
@ timer
def old_getSpineAngle(spineLine: LineString):
    """ Return the angle of the spine Line by using the anchor point and the spine point
    
    Old Idea:
        Return the angle between the two Lines
        Line 1: The line formed between the spine head and the anchor point
        Line 2: The line formed by two points on the segment tracing. 
        Grab two points, one “up” and the other “down” the segment from the spine anchor point. 
        I think the anchor point on the segment tracing is our new “position”.

    Args:
        deprecated - segmentLine: segment in the for of a LineString
        spineLine: Linestring of spine head to anchor point
    """
    spineLineCoord0 = Point(spineLine.coords[0])
    spineLineCoord1 = Point(spineLine.coords[1])
    sl0x = spineLineCoord0.x
    sl0y = spineLineCoord0.y
    sl1x = spineLineCoord1.x
    sl1y = spineLineCoord1.y

    # abj: 6/24
    dx = sl1x - sl0x
    dy =  sl1y - sl0y

    # Angle between p1 and p2 in radians
    angle_rad = math.atan2(dy, dx)
    angle_deg = angle_rad*180/PI

    # # Range: 0 - 360
    # # Check for Negative angle and add 360 degrees to determine counter clockwise value
    if angle_deg < 0:
        angle_deg = angle_deg + 360 

    # print("m1", m1, "m2", m2, "degree:", angle_deg)
    # return angle_deg
    return angle_rad

def get_angle(line1, line2):

    #segment
    line1Coord0 = Point(line1.coords[0]) # beginning of segment
    line1Coord1 = Point(line1.coords[-1]) # end of segment
    l1x0 = line1Coord0.x
    l1y0 = line1Coord0.y
    l1x1 = line1Coord1.x
    l1y1 = line1Coord1.y

    #spineline
    line2Coord0 = Point(line2.coords[0]) # anchor
    line2Coord1 = Point(line2.coords[-1]) # spine point
    l2x0 = line2Coord0.x
    l2y0 = line2Coord0.y
    l2x1 = line2Coord1.x
    l2y1 = line2Coord1.y

    # Get directional vectors
    d1 = (l1x1 - l1x0, l1y1 - l1y0)
    d2 = (l2x1 - l2x0, l2y1 - l2y0)
    # Compute dot product
    p = d1[0] * d2[0] + d1[1] * d2[1]
    # Compute norms
    n1 = math.sqrt(d1[0] * d1[0] + d1[1] * d1[1])
    n2 = math.sqrt(d2[0] * d2[0] + d2[1] * d2[1])
    # Compute angle
    ang = math.acos(p / (n1 * n2))
    # Convert to degrees if you want
    ang = math.degrees(ang)
    return ang

@ timer
def getSpineAngle(segmentLine: LineString, spineLine: LineString):
    """ 
        Return the angle between the two Lines
        Line 1: The line formed between the spine head and the anchor point
        Line 2: The line formed by two points on the segment tracing. 
    
    Previous Idea:
        Return the angle of the spine Line by using the anchor point and the spine point

    Args:
        segmentLine: segment in the for of a LineString
        spineLine: Linestring of spine head to anchor point
    """
    angle = get_angle(segmentLine, spineLine)
    # print("angle", angle)

    return angle

# abj
@timer
def calculateSegmentOffset(segmentLine: LineString, radiusOffset : int, isPositive: bool):
    # radiusOffset = segmentRadius
    if isPositive:
        offsetSign = 1
    else: 
        offsetSign = -1
    offsettedSegment = shapely.offset_curve(segmentLine, distance = radiusOffset * offsetSign, 
                                            # quad_segs = 16,
                                            join_style = "mitre"
                                            , mitre_limit = 15
                                            )
          
    return offsettedSegment

# abj
def getRunningDistance(segmentLine: LineString):
    """

    Return:
        List (same length as inputted segmentLine), that has the running sum distance at each point in the Line.
    """
    # logger.info(f"segmentLine {segmentLine}")
    # segmentLine = list(segmentLine)
    # logger.info(f"segmentLine {segmentLine}")
    x, y = segmentLine.xy
    runningDistanceList = []
    # print(x)
    currentSum = 0
    prevPoint = Point(x[0], y[0]) # First point
    # nextPoint = Point()
    for i, val in enumerate(x):
        # logger.info(f"i {i} val {val}")
        currentPoint = Point(x[i], y[i])

        currentLineString = LineString([prevPoint, currentPoint])
        currentSum = currentSum + currentLineString.length
        runningDistanceList.append(currentSum)

        prevPoint = currentPoint # keep track of previous point
    # print("runningDistanceList", runningDistanceList)
    return runningDistanceList


@timer
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


def clipLines(series: gp.GeoSeries, zRange: Tuple[int, int]) -> gp.GeoSeries:
    # TODO: vectorized
    return series.apply(clipLine, zRange=zRange)


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
            segment.append((p1[0], p1[1]))

            if not z2InRange:
                # The segment exits the bounds
                point = interpolateAcross(z_min, z_max, p1, line.coords[i+1])
                segment.append(point)
                pushLine(segment, lines)
                segment = []

            continue

        p2 = line.coords[i+1]
        if len(segment) != 0:
            pushLine(segment, lines)
            segment = []

        if z2InRange:
            # The segment enters the bounds
            point = interpolateAcross(z_min, z_max, p2, p1)
            segment.append(point)
        elif (p1[2] < z_min and p2[2] > z_max) or (p2[2] < z_min and p1[2] > z_max):
            # The segment crosses the z bounds; clip and include both parts
            pushLine([interpolate(p1, p2, z_min),
                     interpolate(p1, p2, z_max)], lines)

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
