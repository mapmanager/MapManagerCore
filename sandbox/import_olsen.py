import os
from pprint import pprint
from typing import Tuple
import dataclasses

import numpy as np

import nd2
import roifile

import matplotlib.pyplot as plt

import pandas as pd
from mapmanagercore import MapAnnotations, MultiImageLoader

from mapmanagercore.logger import logger

""" having trouble after install of either nd2 or roifile?

conda create -y -n mmc-env python=3.11
conda activate

# now includes nd2
pip install -e .
# zarr == 3.0.1
# 20250130 need zarr==2.16

pip install nd2
pip install roifile
pip install matplotlib
"""

"""
ROI line is:

ImagejRoi(
    roitype=ROI_TYPE.POLYLINE,
    version=228,
    top=172,
    left=454,
    bottom=975,
    right=513,
    n_coordinates=8,
    integer_coordinates=numpy.array([
        [  0,   0],
        [ 30,  96],
        [ 22, 166],
        [ 56, 316],
        [ 58, 492],
        [ 34, 608],
        [ 38, 758],
        [ 36, 802]], dtype=int32),
)
"""
@dataclasses.dataclass
class OlsenRaw:
    imgData: np.ndarray
    imgDataMax: np.ndarray
    channelOrder: Tuple[str]
    """Order of channels/dimensions from (T, C, Z, Y, x)"""
    voxelSize: Tuple[float, float, float]
    """Voxel size (um) for each dimension in channelOrder"""

    # lines and spines
    linePoints: np.ndarray
    spinePoints: np.ndarray
    spineOrder: str

    # mmMap : int  # : MapAnnotations

def LoadOlsen(nd2Path) -> OlsenRaw:
    """
    Animal_145_Slice_1_Left.nd2
    """
    baseName, _ = os.path.splitext(nd2Path)
    linePath = baseName + '.roi'  # roitype=roifile.ROI_TYPE.POLYLINE
    spinePath = baseName + '_Counts.roi'  # roitype=roifile.ROI_TYPE.POINT

    # image data
    with nd2.ND2File(nd2Path) as myfile:
        voxelSize = myfile.voxel_size()  # List[float, float, float]
        # logger.info(f'myfile.sizes:{myfile.sizes}')
        channelOrder = tuple(myfile.sizes.keys())  # tuple('Z", 'Y', 'X')

    imgData = nd2.imread(nd2Path)
    # imgData = imgData / np.max(imgData) * 255  # convert to 8-bit
    imgDataMax = np.max(imgData, axis=0)

    logger.info(f'1 {imgData.shape} {imgData.dtype}')
    
    # lines
    roiLine = roifile.roiread(linePath)
    if roiLine.roitype != roifile.ROI_TYPE.POLYLINE:
        logger.error(f'spine roi has wrong type {roiLine.roitype}')
        logger.error('   expecting roifile.ROI_TYPE.POLYLINE')
    lineArray = np.array(roiLine.coordinates())  # columns (x, y)
    
    # lines are on 2d (x,y), find brightest z
    zArray = np.ndarray(shape=(lineArray.shape[0],), dtype=np.int64)
    for lineIdx, linePoint in enumerate(roiLine.coordinates()):
        # zAll = imgData[:, linePoint[1], linePoint[0]]
        zMaxIndex = np.argmax(imgData[:, linePoint[1], linePoint[0]], axis=0)
        zArray[lineIdx] = zMaxIndex
    lineArray = np.append(lineArray, zArray[:, None], axis=1)

    # spines
    roiSpine = roifile.roiread(spinePath)
    if roiSpine.roitype != roifile.ROI_TYPE.POINT:
        logger.error(f'spine roi has wrong type {roiSpine.roitype}')
        logger.error('   expecting roifile.ROI_TYPE.POINT')

    spineArray = np.array(roiSpine.coordinates())  # (x,y) coordinates
    counter_positions = roiSpine.counter_positions  # z for each point
    # An array with ndim == 1 is implicitly a row, not a column. 
    spineArray = np.append(spineArray, counter_positions[:, None], axis=1)
    spineOrder = ['XYZ']

    olsenRaw = OlsenRaw(
        imgData=imgData,
        imgDataMax=imgDataMax,
        voxelSize=voxelSize,
        channelOrder=channelOrder,
        #
        linePoints=lineArray,
        spinePoints=spineArray,
        spineOrder=spineOrder,
    )
    return olsenRaw

def plotOlsen(olsenRaw : OlsenRaw):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(olsenRaw.imgDataMax)

    xLine = olsenRaw.linePoints[:,0]
    yLine = olsenRaw.linePoints[:,1]
    ax.plot(xLine, yLine, '-ob')

    xSpine = olsenRaw.spinePoints[:,0]
    ySpine = olsenRaw.spinePoints[:,1]
    ax.scatter(xSpine, ySpine, c='r', marker='.')

    plt.show()

def getTimepoint(olsenRaw : OlsenRaw):
    """Make single timepoint with an image."""
    
    loader = MultiImageLoader()
    loader.read(olsenRaw.imgData, channel=0, time=0)
    # loader.read(path=tiffPath, time=0, channel=0)

    # pprint(loader._metadata[0])
    # return

    # Create the annotation map
    map = MapAnnotations(loader,
                         lineSegments=pd.DataFrame(),
                         points = pd.DataFrame())

    olsenRaw.mmMap = map

    return map

def makeSegments(olsenRaw : OlsenRaw):
    linePoints = olsenRaw.linePoints   # (x,y,z) of line points (one segment)
    mmMap = olsenRaw.mmMap

    tp = mmMap.getTimePoint(time=0)
    
    # add segment and segmwent points
    newSegmentID = tp.newSegment()
    for idx, row in enumerate(linePoints):
        x = row[0]
        y = row[1]
        z = row[2]
        logger.info(f'   {idx} appendSegmentPoint segment:{newSegmentID} x:{x} y:{y} z:{z}')
        tp.appendSegmentPoint(newSegmentID, x, y, z)

    logger.info('after appendSegmentPoint appendSegmentPoint')
    print(tp)

    # tp = map.getTimePoint(time=0)
    tp.segments[:]

    return newSegmentID

def addSpines(olsenRaw : OlsenRaw, newSegmentID):
    spinePoints = olsenRaw.spinePoints
    tp = olsenRaw.mmMap.getTimePoint(time=0)

    for row in spinePoints:
        x = row[0]
        y = row[1]
        z = row[2]
        # logger.info(f'   addSpine newSegmentID:{newSegmentID} x:{x} y:{y} z:{z}')
        tp.addSpine(newSegmentID, x=x, y=y, z=z)

    logger.info('after addSpine(s)')
    print(tp)

def makeMap(olsenRaw : OlsenRaw):
    linePoints = olsenRaw.linePoints   # (x,y,z) of line points (one segment)
    spinePoints = olsenRaw.spinePoints
    
    # import mapmanagercore.data
    # tiffPath = mapmanagercore.data.getTiffChannel_1()

    loader = MultiImageLoader()
    loader.read(olsenRaw.imgData, channel=0, time=0)
    # loader.read(path=tiffPath, time=0, channel=0)

    # pprint(loader._metadata[0])
    # return

    # Create the annotation map
    map = MapAnnotations(loader,
                         lineSegments=pd.DataFrame(),
                         points = pd.DataFrame())

    print('1:', map)

    tp = map.getTimePoint(time=0)
    
    # add segment and segmwent points
    newSegmentID = tp.newSegment()
    for row in linePoints:
        x = row[0]
        y = row[1]
        z = row[2]
        logger.info(f'   appending newSegmentID:{newSegmentID} x:{x} y:{y} z:{z}')
        tp.appendSegmentPoint(newSegmentID, x, y, z)

    logger.info('after appendSegmentPoint appendSegmentPoint')
    print(tp)

    # tp = map.getTimePoint(time=0)
    tp.segments[:]

    # logger.info('=== tp segments')
    # print(tp.segments)

    logger.info(f'addSpine(s) shape:{spinePoints.shape}')

    for row in spinePoints:
        x = row[0]
        y = row[1]
        z = row[2]
        # logger.info(f'   addSpine newSegmentID:{newSegmentID} x:{x} y:{y} z:{z}')
        tp.addSpine(newSegmentID, x=x, y=y, z=z)

    logger.info('after addSpine(s)')
    print(tp)

    # save
    savePath = '/Users/cudmore/Desktop/olsen_example.mmap'
    print(f'savePath:{savePath}')
    map.save(savePath)

def loadOlsen():
    from mapmanagercore import MapAnnotations
    savePath = '/Users/cudmore/Desktop/olsen_example.mmap'
    map = MapAnnotations.load(savePath)
    print(map)

    print(map.points[:])
    points = map.points[:]
    print(points.columns)

def plotPlotly(olsenRaw):
    import plotly.graph_objects as go    

    print(olsenRaw.imgData.shape)  # (29, 1568, 1060)
    yMaxPixel = olsenRaw.imgData.shape[1]
    xMaxPixel = olsenRaw.imgData.shape[2]

    # get xyz of segment points
    plotDf = olsenRaw.mmMap.segments['segment'].get_coordinates(include_z=True)    
    plotDf = plotDf.reset_index()
    plotDf = plotDf.reset_index()

    print(plotDf)
    
    import plotly.express as px
    fig = px.scatter(plotDf,
                     x="x",
                     y="y",
                    #  color="species",
                    #  size='petal_length',
                     hover_data='index')

    # scatter = go.Scatter(x=plotDf['x'],
    #                      y=plotDf['y'],
    #                     hover_data=['index'],
    #                     mode='markers+lines')

    # fig = go.Figure(data=scatter)

    fig.update_layout(yaxis_range=[0, yMaxPixel],
                      xaxis_range=[0, xMaxPixel],
                      )

    fig.show()

if __name__ == '__main__':
    # tryROi()

    nd2Path = '/Users/cudmore/Dropbox/data/olson/IKA_A_102 Thy1_Spines_5MeO/Isak_Spines_8_26_23/Animal 145/Animal_145_Slice_1_Left.nd2'
    nd2Path = '/Users/cudmore/Dropbox/data/olson/IKA_A_102 Thy1_Spines_5MeO/Isak_Spines_8_26_23/Animal 145/Animal_145_Slice_1_Right.nd2'
    olsenRaw = LoadOlsen(nd2Path)
    # #plotOlsen(olsenRaw)
    # makeMap(olsenRaw)

    # loadOlsen()

    mmMap = getTimepoint(olsenRaw)
    segmentID = makeSegments(olsenRaw=olsenRaw)
    
    plotPlotly(olsenRaw=olsenRaw)

    # addSpines(olsenRaw=olsenRaw, newSegmentID=segmentID)

    # print(olsenRaw.mmMap)
    # print(olsenRaw.mmMap.points[:])


