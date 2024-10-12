
import csv
import pandas as pd
from mapmanagercore.layers.line import getSpineSide
from mapmanagercore.logger import logger
from mapmanagercore import MapAnnotations, MultiImageLoader
import mapmanagercore
from mapmanagercore.data import getLinesFile, getPointsFile, getTiffChannel_1, getTiffChannel_2
import matplotlib.pyplot as plt

from mapmanagercore.schemas.spine import Spine
def updatePoochDF():
    """ Not Working!

    Attempt to update Spine for "spineSide" the same way that we update columns like "UserType"
    This does not work, and throws an error:
    TypeError: Spine.__init__() got an unexpected keyword argument 'spineSide'
    """
    path = mapmanagercore.data.getSingleTimepointMap()
    map = MapAnnotations.load(path)

    timePoint = 0
    singleTimePoint = map.getTimePoint(timePoint)

    allSpinesDf = singleTimePoint.points[:]
    allSegmentDf = singleTimePoint.segments[:]

    segments = allSegmentDf["segment"]
    print("segments", segments)
    points = allSpinesDf["point"]
    anchor = allSpinesDf["anchor"]
    segmentIDs = allSpinesDf["segmentID"]
    print("segmentIDs", segmentIDs)

    for index, row in enumerate(allSpinesDf["spineSide"]):
        
        _segmentID = segmentIDs[index]
        _segment = segments[_segmentID]
        _spinePoint = points[index]
        _anchor = anchor[index]

        val = getSpineSide(line = _segment, spine = _spinePoint, anchor = _anchor)
        _spine = Spine(spineSide = val)
        
        singleTimePoint.updateSpine(spineId = index, value = _spine)

    allSpinesDf = singleTimePoint.points[:]
    print(allSpinesDf["spineSide"])

def forceUpdate():
    """
        Note: working but slow because we are calling updates twice

        Using dependencies of spine to force updating
            - moves all spine to different position to force update
            and recalculate all the columns that use spine points (including spineSide)
            - moves spines back to maintain original position
    """
    path = mapmanagercore.data.getSingleTimepointMap()
    map = MapAnnotations.load(path)
    timePoint = 0
    singleTimePoint = map.getTimePoint(timePoint)

    allSpinesDf = singleTimePoint.points[:]
    # allSegmentDf = singleTimePoint.segments[:]

    xVal = allSpinesDf["x"]
    yVal = allSpinesDf["y"]
    zVal = allSpinesDf["z"]

    # # Try to force update using move spine with same coordinates
    # # Does not work
    for index, row in enumerate(allSpinesDf["spineSide"]):
        
        x = xVal[index]
        y = yVal[index]
        z = zVal[index]
        singleTimePoint.moveSpine(spineId = index, x=0, y=y, z=z) # different value to force update
        singleTimePoint.moveSpine(spineId = index, x=x, y=y, z=z) # revert back to original position

    allSpinesDf = singleTimePoint.points[:]
    print(allSpinesDf["spineSide"])

    map.old_file_save("C:/Users/johns/Documents/TestMMCMaps/rr30a_s0u_newSpineAngle.mmap")


if __name__ == '__main__':
    # updatePoochDF()
    forceUpdate()











