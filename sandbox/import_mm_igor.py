"""Scripts to import a Map Manager Igor stack and map.

This assumes you have
PyMapManager-Data folder at same level as MapManagerCore folder
(not inside it)

Set `oneTimePoint` to None to make a map with 8 timepoints (hard-coded).

Run this from command line from MapManagerCore like:

    python sandbox/import_mm_igor.py
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List
from shapely.geometry import LineString
from shapely.ops import linemerge

import tifffile

from mapmanagercore import MapAnnotations, MultiImageLoader, MMapLoader
from mapmanagercore.loader.imageio import _createMetaData
from mapmanagercore.logger import logger, setLogLevel

def importStack(folder, timepoints : List[int] = None):
    """
    folder : str
        Path to Igor folder
    oneTimepoint : int | None
        If int specified then make a single timepoint map
            o.w. make a map with numSessions
    """
    
    logger.info(f'folder:{folder}')
    
    _startTime = time.time()

    # hard coded for map rr30a
    numChannels = 2
    numSessions = 8
    maxSlices = 80
        
    # oneTimepoint = 0  # set to None to make a map of numSessions
    if timepoints is None:
        sessionList = range(numSessions)
    else:
        sessionList = timepoints

    mapName = os.path.split(folder)[1]

    igorDict = {
        'dfPoints': [],
        'dfLines': [],
        'imgCh1': [],
        'imgCh2': [],
        'metadata': None,
        'dfGeoLineList': []
    }

    for _idx, sessionID in enumerate(sessionList):

        sessionFolder = f'{mapName}_s{sessionID}'

        linesFile = f'{mapName}_s{sessionID}_la.txt'  # rr30a_s0_la
        linesPath = os.path.join(folder, sessionFolder, linesFile)
        
        pointsFile = f'{mapName}_s{sessionID}_pa.txt'  # rr30a_s0_pa
        pointsPath = os.path.join(folder, sessionFolder, pointsFile)
        
        fileImg1 = f'{mapName}_s{sessionID}_ch1.tif'  #rr30a_s0_ch1'
        pathImg1 = os.path.join(folder, fileImg1)

        fileImg2 = f'{mapName}_s{sessionID}_ch2.tif'  #rr30a_s0_ch2'
        pathImg2 = os.path.join(folder, fileImg2)

        #
        # load point and line annotations
        dfPoints = pd.read_csv(pointsPath, header=1)
        dfLines = pd.read_csv(linesPath, header=1)

        _imgData1 = tifffile.imread(pathImg1)
        if _imgData1.shape[0] < maxSlices:
            # padd
            _numNew = maxSlices - _imgData1.shape[0]
            _newSlices = np.zeros((_numNew, _imgData1.shape[1], _imgData1.shape[2]), dtype=_imgData1.dtype)
            _imgData1 = np.concatenate((_imgData1, _newSlices), axis=0)

        _imgData2 = tifffile.imread(pathImg2)
        if _imgData2.shape[0] < maxSlices:
            # padd
            _numNew = maxSlices - _imgData2.shape[0]
            _newSlices = np.zeros((_numNew, _imgData2.shape[1], _imgData2.shape[2]), dtype=_imgData2.dtype)
            _imgData2 = np.concatenate((_imgData2, _newSlices), axis=0)

        igorDict['dfPoints'].append(dfPoints)
        igorDict['dfLines'].append(dfLines)
        igorDict['imgCh1'].append(_imgData1)
        igorDict['imgCh2'].append(_imgData2)

        if _idx == 0:
            _metaData = _createMetaData(_imgData1, maxSlices=maxSlices, numChannels=numChannels)
            igorDict['metadata'] = _metaData

        ##
        # lines
        ##
        segments = dfLines['segmentID'].unique()
        for segmentID in segments:
            dfSegment = dfLines[ dfLines['segmentID']==segmentID ]

            # TODO: reduce resolution of line by removing every other point?
            
            n = len(dfSegment)
            zxy = np.ndarray((n,3), dtype=int)
            zxy[:,0] = dfSegment['x'].to_list()
            zxy[:,1] = dfSegment['y'].to_list()
            zxy[:,2] = dfSegment['z'].to_list()

            # "t","segmentID","segment","radius","modified"
        
            lineString = LineString(zxy)

            # print(segmentID, 'n:', n, 'lineString len:', lineString.length, len(lineString.coords))

            geoDict = {}
            geoDict['t'] = _idx  # !!!
            geoDict['segmentID'] = segmentID
            geoDict['segment'] = lineString
            geoDict['radius'] = 4
            geoDict['modified'] = 0

            # dfGeoList.append(geoDict)
            igorDict['dfGeoLineList'].append(geoDict)

    #
    dfGeoLine = pd.DataFrame(igorDict['dfGeoLineList'])
    dfGeoPoints = pd.DataFrame()

    # print('dfGeoLine:')
    # print(dfGeoLine)

    # make a loader with all segments and no spines
    loader = MultiImageLoader(
        lineSegments=dfGeoLine,
        points=dfGeoPoints,
        # metadata=igorDict['metadata']
        )

    # append all images to the loader
    logger.info('appending images')
    for _idx, sessionID in enumerate(sessionList):
        # logger.info(f'appending image time= _idx:{_idx}')
        # print('   ch1:', igorDict['imgCh1'][_idx].shape)
        # print('   ch2:', igorDict['imgCh2'][_idx].shape)
        loader.read(igorDict['imgCh1'][_idx], time=_idx, channel=0)
        loader.read(igorDict['imgCh2'][_idx], time=_idx, channel=1)
        loader.readMetadata(igorDict['metadata'], time=_idx)

    #
    # make map from loader
    logger.info('making map from loader')
    map = MapAnnotations(loader)

    #
    # add all spines to the loader
    import warnings
    warnings.filterwarnings("ignore")

    # list of dict, one dict per session
    # each dict maps original igor spine id to core spine id
    # use this at end to load, convert and then connect spines with igor map
    # sessionMapList = []

    _totalAdded = 0
    for _idx, sessionID in enumerate(sessionList):
        # _idx is t in core, sessionID is from igor

        _oneTimepoint = map.getTimePoint(_idx)

        _t = _idx
        dfPoints = igorDict['dfPoints'][_idx]
        
        # original has roiType in ['globalPIvot' 'pivotPnt' 'controlPnt' 'spineROI']
        # print('unique:', dfPoints['roiType'].unique())
        
        dfSpines = dfPoints[ dfPoints['roiType']=='spineROI' ]

        # each session needs to map igor spine id to core spine id
        # thisSessionMap = {}  #np.zeros((len(dfSpines)))

        _n = len(dfSpines)
        print('=== ADDING SPINE:', 'core t:', _idx, 'igor-sessionID:', sessionID, 'n=', _n)
      
        _count = 0
        for index, row in dfSpines.iterrows():
            # index is the original row before reducing to spineROI
            segmentID = int(row['segmentID'])
            x = row['x']
            y = row['y']
            z = row['z']
            
            # print('   ADDING SPINE:', 'sessionID:', sessionID, '_count', _count, 'index:', index, 'segmentID:', segmentID, 't:', _t, 'x/y/z', x, y, z)

            # TODO: add spine does not connect anchor properly
            # newSpineID = _oneTimepoint.addSpine(segmentId=(segmentID,_t),
            newSpineID = _oneTimepoint.addSpine(segmentId=segmentID,
                                    x=x, y=y, z=z,
                                    #brightestPathDistance=brightestPathDistance,
                                    #channel=channel,
                                    #zSpread=zSpread
                                    )
            
            # print(f'   newSpineID:{newSpineID}')

            # original spine row is `index`, in core will be newSpineID
            # map to spine id to igor spine id
            # thisSessionMap[index] = newSpineID
            
            _count += 1

        # sessionMapList.append(thisSessionMap)

        _totalAdded += _count
        # print(f'_idx:{_idx} sessionID:{sessionID} add {_count} spines')

    print('total added spine:', _totalAdded)
    
    # computer all spine and segment columns (e.g. intensity analysis)
    print('map.points[:]')
    map.points[:]
    print('map.segmentsef[:]')
    map.segments[:]

    #
    # save our new map
    if timepoints is None or len(timepoints)>1:
        mmMapSessionFile = f'rr30a.mmap'
    else:
        oneTimepoint = timepoints[0]
        mmMapSessionFile = f'rr30a_s{oneTimepoint}.mmap'
    savePath = os.path.join('sandbox', 'data', mmMapSessionFile)
    logger.info(f'saving: {savePath}')
    map.save(savePath)

    # make sure we can load the map
    logger.info('re-load as map2')
    map2 = MapAnnotations(MMapLoader(savePath))
    logger.info(f'map2:{map2}')

    _stopTime = time.time()
    print(f'done, import took {round(_stopTime-_startTime,3)} s')

def convertRunMap(igorFolder, coreMapPath = 'sandbox/data/rr30a.mmap'):
    """Convert run map saved in xxx to core centric spine ID.
    """

    print('loading coreMapPath:', coreMapPath)
    # coreMap = MapAnnotations(MMapLoader(coreMapPath).cached())
    coreMap = MapAnnotations(MMapLoader(coreMapPath))
    print(coreMap)
    
    #
    # load run/seg map manually exported from PyMapManager mmMap
    igorRunMapPath = 'igorRunMapExport.npy'
    igorRunMapPath = os.path.join(igorFolder, igorRunMapPath)
    igorRunMap = np.load(igorRunMapPath)
    print('igorRunMap:', igorRunMap.shape)
    nSessions = igorRunMap.shape[1]
    nRows = igorRunMap.shape[0]

    print(igorRunMap[0:5,:])
    print(igorRunMap[:-5,:])
    
    igorSegMapPath = 'igorSegMapExport.npy'
    igorSegMapPath = os.path.join(igorFolder, igorSegMapPath)
    igorSegMap = np.load(igorSegMapPath)
    print('igorSegMap:', igorSegMap.shape)
    print(igorSegMap)
    nSegRows = igorSegMap.shape[0]

    # sandbox/data/rr30a.mmap

    return

    # load the map
    coreMapPath = '/Users/cudmore/Sites/MapManagerCore/sandbox/data/rr30a.mmap'
    map = MapAnnotations(MMapLoader(coreMapPath).cached())

    for t in range(nSessions-1):
        for segRow in range(nSegRows):
            thisSeg = int(igorSegMap[segRow, t])
            nextSeg = int(igorSegMap[segRow, t+1])
            print('t:', t, 'thisSeg:', thisSeg, 'nextSeg:', nextSeg)
            map.connectSegment(segmentKey=(thisSeg,t), toSegmentKey=(nextSeg, t+1))

def loadForSpineDist():
    path = 'sandbox/data/rr30a_s0.mmap'
    map = MapAnnotations(MMapLoader(path))
    
    print(map.points.columns)
    map.points.columnsAttributes

    print(map.points[['segmentID', 'userType', 'point', 'spineDistance', 'spineSide']])

if __name__ == '__main__':
    setLogLevel()
    folder = '../PyMapManager-Data/maps/rr30a'
    
    # 6 and 7 have error
    oneTimepoint = [0,1]  # 1 timepoint (from igor)
    # oneTimepoint = None  # 8 session map
    
    # 1) works
    importStack(folder, timepoints=oneTimepoint)

    # 2) load multi tp core map (created in xxx) and convert spine IDs
    # from igor to core spine id
    # convertRunMap(folder)

    # 3)
    # loadForSpineDist()
