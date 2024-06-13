from typing import Optional
import numpy as np
from shapely.geometry import LineString, Point
import shapely
import pandas as pd
import geopandas as gpd
import shapely.geometry
from shapely.geometry.base import BaseGeometry
from .benchmark import timer
import itertools


@timer
def filterMask(d: pd.Index, index_filter: list):
    """Filter a mask based on a list of indices.

    Args:
        d (pd.Index): Mask to filter
        index_filter (set or list-like): List of indices to filter

    Returns:
        pd.Series: Filtered mask
    """
    if index_filter == None or len(index_filter) == 0:
        return np.full(len(d), False)

    return ~d.isin(index_filter)


def generateGrid(stepX: int, stepY: int, points: int):
    """Generate a grid of points.

    Args:
        stepX (int): Step size in the x direction
        stepY (int): Step size in the y direction
        points (int): Number of points across each axis of the grid

    Returns:
        pd.DataFrame: DataFrame with columns x and y representing points on the grid
    """
    distanceX = stepX * points
    distanceY = stepY * points
    x = np.arange(0, distanceX, stepX) - (stepX * ((distanceX * 0.5) // stepX))
    y = np.arange(0, distanceY, stepY) - (stepY * ((distanceY * 0.5) // stepY))
    return pd.DataFrame(itertools.product(x, y), columns=["x", "y"])


def shapeGrid(shape: BaseGeometry, points: int, overlap=0):
    """Generate a grid of offsets using a shape as distance.

    Args:
        shape (BaseGeometry): Shape to generate the grid with
        points (int): Number of shapes across each axis of the grid
        overlap (float, optional): The shape's overlap percentage on the grid. Defaults to 0.

    Returns:
        pd.DataFrame: DataFrame with columns x and y representing points on the grid
    """
    minx, miny, maxx, maxy = shape.bounds
    width = maxx - minx
    height = maxy - miny
    overlap = 1 - overlap
    return generateGrid(width * overlap, height * overlap, points)

def set_precision(series: gpd.GeoSeries, *args, **kwargs):
    """Set the precision of a GeoSeries."""
    return gpd.GeoSeries(shapely.set_precision(series.values, *args, **kwargs), series.index, series.crs)


def force_2d(series: gpd.GeoSeries, *args, **kwargs):
    """Force a GeoSeries shapes to 2D."""
    return gpd.GeoSeries(shapely.force_2d(series.values, *args, **kwargs), series.index, series.crs)


def count_coordinates(series: gpd.GeoSeries, *args, **kwargs):
    """Count the number of coordinates in each row of a GeoSeries."""
    return pd.Series(shapely.get_num_coordinates(series.values, *args, **kwargs), series.index, series.crs)


def union(a: gpd.GeoSeries, b: gpd.GeoSeries, grid_size: int):
    """Union the shapes of corresponding row of two GeoSeries."""
    return gpd.GeoSeries(shapely.union_all([a, b], axis=0, grid_size=grid_size), a.index, a.crs)


def injectPoint(line: LineString, point: Point):
    """Inject a point into a line.

    Args:
        line (LineString): Line to inject the point into
        point (Point): Point to inject into the line

    Returns:
        LineString: Line with the point injected
        int: Index of the injected point within the line
    """
    # get the distance of the point along the line
    distance = line.project(point)
    currentPosition = 0.0
    coords = line.coords

    for i in range(len(coords) - 1):
        point1 = coords[i]
        point2 = coords[i + 1]
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        dz = point1[2] - point2[2]
        segment_length = (dx**2 + dy**2 + dz**2) ** 0.5

        currentPosition += segment_length
        if distance == currentPosition:
            # the point already exists on the line
            return None, None

        if distance <= currentPosition:
            # inject the point into the line
            return LineString([*coords[:i+1], point.coords[0], *coords[i+1:]]), i+1

    # append the point to the end of the line
    return LineString([*coords, point.coords[0]]), len(coords)


def injectLine(line: LineString, newLine: LineString, leftPoint: Optional[Point], rightPoint: Optional[Point]):
    """Inject a line into another line between the leftPoint and rightPoint.

    Args:
        line (LineString): Line to inject the new line into
        newLine (LineString): Line to inject into the line
        leftPoint (Point, Optional): Point to start injecting the new line. 
            If None, the new line will be appended to the start of the line
        rightPoint (Point, Optional): Point to end injecting the new line.
            If None, the new line will be appended to the end of the line

    Returns:
        LineString: Line with the new line injected
    """

    if not leftPoint and not rightPoint:
        # replace the entire line
        return newLine

    if len(newLine.coords) > 0:
        # check if the new line does not have the start point
        if leftPoint and newLine.coords[0] != leftPoint.coords[0]:
            # prepend the start point to the new line
            newLine = LineString([leftPoint.coords[0], *newLine.coords])

        # check if the new line does not have the end point
        if rightPoint and newLine.coords[-1] != rightPoint.coords[0]:
            # append the end point to the new line
            newLine = LineString([*newLine.coords, rightPoint.coords[0]])

    # get the distance of the points along the line
    startDistance = line.project(leftPoint) if leftPoint else None
    endDistance = line.project(rightPoint) if rightPoint else None

    currentPosition = 0.0
    coords = line.coords
    startIdx = None
    endIdx = len(coords)

    # find the start and end index of the line
    for i in range(len(coords) - 1):
        point1 = coords[i]
        point2 = coords[i + 1]
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        dz = point1[2] - point2[2]
        segment_length = (dx**2 + dy**2 + dz**2) ** 0.5

        currentPosition += segment_length
        if startDistance and startIdx is None and startDistance <= currentPosition:
            startIdx = i + 1
        if endDistance != None and endDistance <= currentPosition:
            endIdx = i + 1
            break

    if not leftPoint:
        # append the new line/Point to the start of the line
        if len(newLine.coords) == 0:
            return LineString([leftPoint.coords[0], *coords[endIdx:]])
        return LineString([*newLine.coords, *coords[endIdx:]])

    if not rightPoint:
        # append the new line/Point to the end of the line
        if len(newLine.coords) == 0:
            return LineString([*coords[:startIdx], leftPoint.coords[0]])
        return LineString([*coords[:startIdx], *newLine.coords])

    startIdx = startIdx or 0

    # inject the new line into the line
    return LineString([*coords[:startIdx], *newLine.coords, *coords[endIdx:]])
