"""Test a 2D synthetic image.
"""
from enum import Enum
from pprint import pprint
import dataclasses
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go    
import plotly.express as px

from mapmanagercore import MapAnnotations, MultiImageLoader
from mapmanagercore.logger import logger

class SinType(Enum):
    horizontal = 1
    vertical = 2
    diagonal = 3

def sinImage(sinType:SinType = SinType.horizontal):
    """sin image.

    see https://stackoverflow.com/questions/57534808/generate-an-image-of-a-sloped-sinewave
    """
    # M = 1000  # TODO make it a rectangle, not sqaure
    N = 1000
    x = np.linspace(-np.pi,np.pi, N)
    baseInt = 127.0
    sineAmp = 127.0
    numBands = 3  # 8,0
    sine1D = baseInt + (sineAmp * np.sin(x * numBands))
    sine1D = np.uint8(sine1D)
    
    # plt.plot(sine1D)
    # plt.show()
    # return

    sine2D = np.tile(sine1D, (N,1))
    if sinType==SinType.horizontal:
        sine2D = np.rot90(sine2D)

    elif sinType==SinType.diagonal:
        sine2D = np.ndarray((N,N), dtype=np.uint8)
        _angle = 1  # 1 is diagonal
        for i in range(N):
            sine2D[i]= np.roll(sine1D,-i*_angle)  # shift the 1D sin data by -i, -i increases with rows

    return sine2D

def addSpines(mmMap : MapAnnotations, segmentID: int):
    if segmentID == 1:
        spinePoints = [
            [100, 200],
            [200, 600],
            [300, 200],
            [400, 600],
            [500, 200],
        ]
    elif segmentID == 2:
        spinePoints = [
            [650, 700],
            [700, 800],
            [750, 700],
            [800, 800],
            [850, 700],
        ]

    z = 0  # 2d images require z=0 (one image slice)

    tp = mmMap.getTimePoint(time=0)

    for idx, row in enumerate(spinePoints):
        x = row[0]
        y = row[1]
        tp.addSpine(segmentID, x, y, z)

def addSegment(mmMap : MapAnnotations, segmentID: int):
    tp = mmMap.getTimePoint(time=0)
    
    if segmentID == 1:
        linePoints = [
            [100, 400],
            [200, 400],
            [300, 400],
            [400, 400],
            [500, 400],
            [600, 400],
            [700, 400],
        ]
    elif segmentID == 2:
        linePoints = [
            [600, 750],
            [650, 750],
            [700, 750],
            [750, 750],
            [800, 750],
            [850, 750],
            [900, 750],
        ]

    z = 0  # 2d images require z=0 (one image slice)

    newSegmentID = tp.newSegment()
    for idx, row in enumerate(linePoints):
        x = row[0]
        y = row[1]
        logger.info(f'   {idx} appendSegmentPoint segment:{newSegmentID} x:{x} y:{y}')
        tp.appendSegmentPoint(newSegmentID, x, y, z)

    return newSegmentID

def makeMap():
    """Make single timepoint with a synthetic image."""
    
    imgData = sinImage(SinType.horizontal)

    loader = MultiImageLoader()
    loader.read(imgData, channel=0, time=0)

    # Create the annotation map
    mmMap = MapAnnotations(loader,
                         lineSegments=pd.DataFrame(),
                         points = pd.DataFrame())

    print('metadata is:')
    pprint(mmMap.loader.metadata(t=0))

    return mmMap

@dataclasses.dataclass
class SpinePlotQt:
    """Holds x/y coordinates to plot spines and spine lines.
    """
    numSpines: int
    spinePointID: List[int]
    # point
    xSpinePoint: List[int]
    ySpinePoint: List[int]
    # lines
    xSpineLine: List[int]
    ySpineLine: List[int]

def _getSpinePlot_Qt(mmMap: MapAnnotations, timepoint: int) -> SpinePlotQt:
    """Get x/y coordinates to plot spines and spine lines.
    
    Arguments:
        mmMap: The map.
        tp: The timepoint.
    """
    tp = mmMap.getTimePoint(time=timepoint)  # AnnotationsLayers

    # each spine has a column anchorX and anchorY
    # _points = tp.points[:]
    # print(_points)
    # print(f'tp.points.columnsAttributes: {tp.points.columnsAttributes}')
    # for k, v in tp.points.columnsAttributes.items():
    #     print(f"{k}: {v['description']}")

    # v2
    points = tp.points[:]
    points = points.reset_index()  # move segmentID label into column
    points = points.reset_index()  # move spineID label into column
    print(f'points df is: {type(points)}')
    print(points)

    # print(f'points columns are:')
    # for columns in points.columns:
    #     print(f'   {columns}')

    #
    # v1
    # 
    # anchorLine is a LINESTRING for each spine from (anchor to head)
    # anchorLine does not return segmentID !!!
    # abb use tp.points[:] anchorX and anchorY
    spineLines = tp.points['anchorLine']
    # print('anchorLine:')
    # print(spineLines)
    spineLines = spineLines.get_coordinates(include_z=True)
    spineLines = spineLines.reset_index()  # move spineID label into column

    # print('spineSide:')
    # print(tp.points['spineSide'])

    numSpines = len(spineLines['spineID'].unique())
    spinePointID = [None] * numSpines  # keep track of each spineID label in our plot
    xSpinePoint = [None] * numSpines
    ySpinePoint = [None] * numSpines
    xSpineLine = [None] * (numSpines * 2)
    ySpineLine = [None] * (numSpines * 2)
    for _idx, spineLabel in enumerate(spineLines['spineID'].unique()):
        oneLine = spineLines[spineLines['spineID']==spineLabel]  # 2 rows (anchor, head)
        
        # spineID in df is np.float64
        spinePointID[_idx] = int(oneLine.iloc[0]['spineID'])

        # spine points
        xSpinePoint[_idx] = oneLine.iloc[1]['x']
        ySpinePoint[_idx] = oneLine.iloc[1]['y']

        # spine lines
        firstRow = _idx * 3
        xSpineLine[firstRow:firstRow+1] = oneLine['x']  # assigns 2 rows (anchor, head)
        ySpineLine[firstRow:firstRow+1] = oneLine['y']
        
        xSpineLine[firstRow+2] = np.nan  # make spine lines disjoint
        ySpineLine[firstRow+2] = np.nan

    _ret = SpinePlotQt(
        numSpines=numSpines,
        spinePointID=spinePointID,
        xSpinePoint=xSpinePoint,
        ySpinePoint=ySpinePoint,
        xSpineLine=xSpineLine,
        ySpineLine=ySpineLine
    )
    return _ret

def plotPlotly(mmMap: MapAnnotations):

    logger.warning('1) check pandas')

    timepoint = 0
    channel = 0
    sliceIdx = 0
    
    spinePlotQt = _getSpinePlot_Qt(mmMap, timepoint)

    # get image(s) from the core ???
    # either this
    # imgData = mmMap._images.fetchSlices(time=0, channel=0, sliceRange=[0,1])
    # or this
    imgData = mmMap._images.loadSlice(time=timepoint, channel=channel, slice=sliceIdx)
    # or this
    # imgData = mmMap.loader._images(t=0, channel=0)
    logger.info(f'imgData:{imgData.shape} {imgData.dtype} {np.max(imgData)}')

    # imageBounds = mmMap.imageBounds(t=timepoint, channel=channel)
    # yMaxPixel = imageBounds[1]
    # xMaxPixel = imageBounds[2]
    # logger.info(f'xMaxPixel:{xMaxPixel} yMaxPixel:{yMaxPixel}')

    # plot segment
    # what is the difference between 'roughTracing' and 'segment'???
    # get xyz of segment points
    tp = mmMap.getTimePoint(time=timepoint)  # AnnotationsLayers
    plotDf = tp.segments['roughTracing'].get_coordinates(include_z=True)    
    plotDf = plotDf.reset_index()  # moves segmentID into column
    plotDf = plotDf.reset_index()  # moves index into column

    fig = go.Figure()

    # segments
    aSegmentLine = go.Scatter(x=plotDf["x"],
                          y=plotDf["y"],
                          mode='markers+lines'
    )
    fig.add_trace(aSegmentLine)

    # spines
    spineScatter = go.Scatter(x=spinePlotQt.xSpinePoint,
                          y=spinePlotQt.ySpinePoint,
                          mode='markers'
    )
    fig.add_trace(spineScatter)

    # spine lines (anchor, head)
    spineLineScatter = go.Scatter(x=spinePlotQt.xSpineLine,
                          y=spinePlotQt.ySpineLine,
                          connectgaps=False,
                          mode='markers+lines'
    )
    fig.add_trace(spineLineScatter)

    # Add image(s)
    fig.add_trace(
            go.Heatmap(z=imgData, colorscale='greens')
    )

    # fig.update_layout(yaxis_range=[0, yMaxPixel],
    #                   xaxis_range=[0, xMaxPixel],
    #                   )

    fig.show()

def run():
    mmMap = makeMap()
    
    segmentID = 1
    newSegmentID = addSegment(mmMap, segmentID)
    addSpines(mmMap, newSegmentID)

    segmentID = 2
    newSegmentID = addSegment(mmMap, segmentID)
    addSpines(mmMap, newSegmentID)

    # from pprint import pprint
    # pprint(mmMap.points.columnsAttributes)
    
    # AHA anchor line is a LINESTRING for each spine from (anchor to head)
    # logger.warning('anchorLine is:')
    # print(mmMap.points['anchorLine'])
    
    plotPlotly(mmMap)

if __name__ == '__main__':
    # sin = sinImage(SinType.horizontal)
    # sin = sinImage(SinType.vertical)
    # # sin = sinImage(SinType.diagonal)
    # plt.imshow(sin, cmap='gray')
    # plt.show()

    run()