import math
from typing import Union, List
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
import shapely
import skimage.draw
from mapmanagercore.config import Segment, Spine
from mapmanagercore.layers.utils import dropZ

from mapmanagercore.logger import logger

def filterMask(d, index_filter):
    if index_filter == None or len(index_filter) == 0:
        return np.full(len(d), False)
    return ~d.isin(index_filter)


def shapeIndexes(d: Union[Polygon, LineString]):
    d = dropZ(d)
    # abb 20240509
    # sometimes ROIs are split in 2, this is a cludge taking the first [0] polugon
    # for spine roi, we want the one that contains the spine point
    # if isinstance(d, Polygon):
    if isinstance(d, (Polygon, MultiPolygon)):
        if isinstance(d, MultiPolygon):
            d = d.geoms[0]
        x, y = zip(*d.exterior.coords)
        return skimage.draw.polygon(x, y)

    x, y = zip(*d.coords)
    return skimage.draw.line(int(x[0]), int(y[0]), int(x[1]), int(y[1]))


def validateColumns(values: dict[str, any], typeColumns: Union[Spine, Segment]):
    typeColumns = typeColumns.__annotations__
    for key, value in values.items():
        if not key in typeColumns:
            raise ValueError(f"Invalid column {key}")
        expectedType = typeColumns[key]
        if not isinstance(value, expectedType):
            try:
                values[key] = expectedType(value)
                return
            except:
                raise ValueError(f"Invalid type for column {key}")


def polygonUnion(a: Polygon, b: Polygon) -> Polygon:
    shape = shapely.union(a, b, grid_size=1)
    if isinstance(shape, GeometryCollection):
        return next(s for s in shape.geoms if isinstance(s, Polygon))
    return shape

# abb
def findBrightestIndex(x, y, z,
                        xyzLine : List[List[float]],
                        image: np.ndarray,
                        numPnts: int = 5,
                        lineWidth: int = 1) -> int:
    """Find the brightest path in an image volume
        from a point (x,y,z) to a line (xyzLine).
    
    Parameters
    ----------
    x, y, z
        coordinate of the point (spine)
    xyzLine
        xyz points of the line 
    image: np.array
        2D image data
    numPnts
        Parameter for the search, seach +/- from closest point (seed point)
    lineWidth
        Width of line to calculate each candidate intensity profile

    Returns
    -------
    Index on the line which has the brightest path from point to line
    """

    closestIndex = findClosestIndex(x, y, z, xyzLine)
    
    firstPoint = closestIndex-numPnts
    lastPoint = closestIndex+numPnts
    
    if(firstPoint < 0):
        firstPoint = 0
        
    if(lastPoint > len(xyzLine)):
        lastPoint = len(xyzLine) - 1
    
    # Grab a list of candidate points on the line, loop through temp
    candidatePoints = xyzLine[firstPoint:lastPoint]
  
    brightestIndex = None
    brightestSum = -1
    
    for index, candidatePoint in enumerate(candidatePoints):
        sourcePoint = np.array([x, y])
        # sourcePoint = np.array([y, x])

        destPoint = np.array([candidatePoint[0], candidatePoint[1]])
        # destPoint = np.array([candidatePoint[1], candidatePoint[2]])

        candidateProfile = skimage.measure.profile_line(image, sourcePoint, destPoint, lineWidth)
        oneSum = np.sum(candidateProfile)
        
        if oneSum > brightestSum:
            brightestSum = oneSum
            # Add CurrentIdx to properly offset
            brightestIndex = index 
     
    return brightestIndex + firstPoint

# abb
def findClosestIndex(x, y, z, xyzLine : List[List[float]]) -> int:
    """Find the closest point to (x,y,z) on line.
    """
    dist = float('inf')
    closestIdx = None
    for idx, point in enumerate(xyzLine):
        dx = abs(x - point[0])
        dy = abs(y - point[1])
        dz = abs(z - point[2])
        _dist = math.sqrt( dx**2 + dy**2 + dz**2)
        if _dist < dist:
            dist = _dist
            closestIdx = idx
    return closestIdx

# abb
def polygonToMask(poly : shapely.Polygon, nx : int = 1024, ny : int = 1024) -> np.array:
    """Convert a polygon to a binary mask.
    """
    from matplotlib.path import Path

    poly = np.array(poly.exterior.coords)

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    # y and x's are reversed
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))

    y, x = y.flatten(), x.flatten()

    points = np.vstack((y,x)).T

    polyPath = Path(poly)
    polyMask = polyPath.contains_points(points, radius=0)
    polyMask = polyMask.reshape((ny,nx))
    polyMask = polyMask.astype(int)

    return polyMask

# abb
def _getOffset(distance, numPoints):
    """Get a list of candidate points where mask will be moved.
    
    Parameters
    ----------
    distance
        Distance between points
    numPnts
        Number of points (each side of a square), has to be odd
        
    Returns
    -------
        List of [x,y] offset pixels [[x, y]]
    """
    coordOffsetList = []

    xStart = - (math.floor(numPoints/2)) * distance
    xEnd = (math.floor(numPoints/2) + 1) * distance

    yStart = - (math.floor(numPoints/2)) * distance
    yEnd = (math.floor(numPoints/2) + 1) * distance

    xList = np.arange(xStart, xEnd, distance)
    yList = np.arange(yStart, yEnd, distance)

    for xPoint in xList:
        for yPoint in yList:
            coordOffsetList.append([xPoint, yPoint])

    return coordOffsetList

# abb
def calculateLowestIntensityOffset(mask, distance, numPoints, img):
    """Get the [x,y] offset of the lowest intensity from a number of candidate positions.

    Parameters
    ----------
    mask
        The mask that will be moved around to check for intensity at various positions
    distance
        How many steps in the x,y direction the points in the mask will move
    numPoints
        (has to be odd) Total number of moves made (total positions that we will check for intensity)

    Returns
    -------
        The [x,y] offset with lowest intensity
    """

    offsetList = _getOffset(distance = distance, numPoints = numPoints)

    lowestIntensity = math.inf
    lowestIntensityOffset = 0
    # lowestIntensityMask = None
    for offset in offsetList:
        
        # logger.info(f'offset: {offset}')
        
        currentIntensity = 0

        _offsetMask = calculateBackgroundMask(mask, offset)
        if _offsetMask is None:
            # given offset is beyond image bounds
            continue
    
        pixelIntensityofMask = img[_offsetMask == 1]

        totalIntensity = np.sum(pixelIntensityofMask)
        currentIntensity = totalIntensity
        if(currentIntensity < lowestIntensity):
            lowestIntensity = currentIntensity
            lowestIntensityOffset = offset
            # lowestIntensityMask = pixelIntensityofMask

    return lowestIntensityOffset

# abb
def calculateBackgroundMask(mask, offset):
    """Offset the values of a given mask and return the background mask
        
    Masks will either be the spine or segment/ dendrite mask.
    """
    maskCoords = np.argwhere(mask == 1)
    backgroundPointList = maskCoords + offset

    # Separate into x and y
    # Construct the 2D mask using the offset background
    backgroundPointsX = backgroundPointList[:,0]
    backgroundPointsY = backgroundPointList[:,1]

    backgroundMask = np.zeros(mask.shape, dtype = np.uint8)

    # logger.info(f"backgroundPointsY:{backgroundPointsY}")
    # logger.info(f"backgroundPointsX:{backgroundPointsX}")
    
    try:
        backgroundMask[backgroundPointsY,backgroundPointsX] = 1
    except (IndexError) as e:
        # Account for out of bounds 
        return None
    
    return backgroundMask

def _testPolygonToMask():
    poly = shapely.Polygon([[5,5], [5,100], [220,320], [250,230]])

    x,y = poly.exterior.xy
    plt.plot(x, y, 'r')

    mask = polygonToMask(poly)
    plt.imshow(mask)

    plt.show()

def _testOffsets():
    distance = 50
    numPoints = 5
    offsetList = _getOffset(distance = distance, numPoints = numPoints)
    
    print('offsetList:', offsetList)

    xPlot = [offset[0] for offset in offsetList]
    yPlot = [offset[1] for offset in offsetList]

    plt.plot(xPlot, yPlot, 'o')
    plt.show()

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # _testPolygonToMask()

    _testOffsets()