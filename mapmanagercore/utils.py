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

from mapmanagercore.logger import logger

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

    try:
        x = np.arange(0, distanceX, stepX) - (stepX * ((distanceX * 0.5) // stepX))
        y = np.arange(0, distanceY, stepY) - (stepY * ((distanceY * 0.5) // stepY))
        return pd.DataFrame(itertools.product(x, y), columns=["x", "y"])
    except (ValueError) as e:
        logger.error(f'points:{points}')
        logger.error(f'distanceX:{distanceX} stepX:{stepX}')
        logger.error(f'distanceY:{distanceX} stepX:{stepY}')
        raise

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
    try:
        _grid = generateGrid(width * overlap, height * overlap, points)
    except (ValueError) as e:
        logger.error(f'shape:{shape}')
        logger.error(f'shape.bounds:{shape.bounds}')
        logger.error(f'minx:{minx} miny:{miny} maxx:{maxx} maxy:{maxy}')
        raise
    
    return _grid

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

def interpolate(lines: gpd.GeoSeries, distance: gpd.GeoSeries):
    """Union the shapes of corresponding row of two GeoSeries."""
    return gpd.GeoSeries(shapely.line_interpolate_point(lines, distance.values), lines.index, lines.crs)

def covered_by(a: gpd.GeoSeries, b: gpd.GeoSeries):
    return pd.Series(shapely.covered_by(a, b), a.index)

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

def getAutoContrast(imgData : np.ndarray) -> tuple[int, int]:
    # https://forum.image.sc/t/macro-for-image-adjust-brightness-contrast-auto-button/37157/5
    # Python rewriting of ImageJ's auto-threshold option (Image > Adjust > Brightness/Contrast > 'Auto' button)
    # Based on https://github.com/imagej/ImageJ/blob/706f894269622a4be04053d1f7e1424094ecc735/ij/plugin/frame/ContrastAdjuster.java#L780
    # (function autoAdjust)
    # The algorithm is basically a contrast setting the max white value to the max of the image (and same for black for
    # min), with some saturation : i.e., it's not the max(min) of the image which is actually used but a lower(higher)
    # value to eliminate the thin "tails" of the histogram and get an output dynamic range which allows for good
    # visualisation of most of the image's pixels (at the expense of a few saturated pixels).
    # While some (most ?) algorithms parametrize this saturation to eliminate a set percentage of pixels,
    # ImageJ's algorithm selects the closest values to the max(min) values whose count are over a certain proportion of the
    # total amount of pixels.
    
    im = imgData

    im_type = im.dtype
    im_min = np.min(im)
    im_max = np.max(im)

    # converting image =================================================================================================

    # case of color image : contrast is computed on image cast to grayscale
    if len(im.shape) == 3 and im.shape[2] == 3:
        # depending on the options you chose in ImageJ, conversion can be done either in a weighted or unweighted way
        # go to Edit > Options > Conversion to verify if the "Weighted RGB conversion" box is checked.
        # if it's not checked, use this line
        # im = np.mean(im, axis = -1)
        # instead of the following
        im = 0.3 * im[:,:,2] + 0.59 * im[:,:,1] + 0.11 * im[:,:,0]
        im = im.astype(im_type)

    # histogram computation =============================================================================================

    # parameters of histogram computation depend on image dtype.
    # following https://imagej.nih.gov/ij/developer/macro/functions.html#getStatistics
    # 'The histogram is returned as a 256 element array. For 8-bit and RGB images, the histogram bin width is one.
    # for 16-bit and 32-bit images, the bin width is (max-min)/256.'
    if im_type in (np.uint8, np.int8):  # abb np.int8
        hist_min = 0
        hist_max = 256
    elif im_type in (np.uint16, np.int16, np.int32):
        # use img min/max
        hist_min = im_min
        hist_max = im_max
    else:
        raise NotImplementedError(f"Not implemented for dtype {im_type}")

    # compute histogram
    histogram = np.histogram(im, bins = 256, range = (hist_min, hist_max))[0]
    bin_size = (hist_max - hist_min)/256

    # compute output min and max bins =================================================================================

    # various algorithm parameters
    h, w = im.shape[:2]
    pixel_count = h * w
    # the following values are taken directly from the ImageJ file.
    limit = pixel_count/10
    const_auto_threshold = 5000
    auto_threshold = 0

    auto_threshold = const_auto_threshold if auto_threshold <= 10 else auto_threshold/2
    threshold = int(pixel_count/auto_threshold)

    # setting the output min bin
    i = -1
    found = False
    # going through all bins of the histogram in increasing order until you reach one where the count if more than
    # pixel_count/auto_threshold
    # while not found and i <= 255:
    while not found and i < 255:
        i += 1

        try:
            count = histogram[i]
        except (IndexError) as e:
            logger.error(f'histogram.shape:{histogram.shape} i:{i} threshold:{threshold} {e}')
            logger.error(f'  hist_min:{hist_min} hist_max:{hist_max} threshold:{threshold} {e}')

        if count > limit:
            count = 0
        found = count > threshold
    hmin = i
    found = False

    # setting the output max bin : same thing but starting from the highest bin.
    i = 256
    while not found and i > 0:
        i -= 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
    hmax = i

    # compute output min and max pixel values from output min and max bins ===============================================
    if hmax >= hmin:
        min_ = hist_min + hmin * bin_size
        max_ = hist_min + hmax * bin_size
        # bad case number one, just return the min and max of the histogram
        if min_ == max_:
            min_ = hist_min
            max_ = hist_max
    # bad case number two, same
    else:
        min_ = hist_min
        max_ = hist_max

    # apply the contrast ================================================================================================
    #imr = (im-min_)/(max_-min_) * 255

    # return imr
    min_ = int(min_)
    max_ = int(max_)

    return min_, max_