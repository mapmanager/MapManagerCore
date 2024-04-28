"""Scripts to import a Map Manager Igor stack and map.

This assumes you have
PyMapManager-Data folder at same level as MapManagerCore folder
(not inside it)

Set `oneTimePoint` to None to make a map with 8 timepoints (hard-coded).

Run this from command line from MapManagerCore like:

    python sandbox/import_mm_igor.py
"""

import os
import numpy as np
import pandas as pd
from shapely.geometry import LineString

import tifffile

from mapmanagercore import MapAnnotations, MultiImageLoader, MMapLoader
from mapmanagercore.loader.imageio import _createMetaData
from mapmanagercore.logger import logger, setLogLevel

def importStack(folder):
    logger.info(f'folder:{folder}')
    
    # hard coded for map rr30a
    numChannels = 2
    numSessions = 8
    maxSlices = 80
        
    oneTimepoint = 0  # set to None to make a map of numSessions
    if oneTimepoint is None:
        sessionList = range(numSessions)
    else:
        sessionList = [0]

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

            n = len(dfSegment)
            zxy = np.ndarray((n,3), dtype=int)
            zxy[:,0] = dfSegment['x'].to_list()
            zxy[:,1] = dfSegment['y'].to_list()
            zxy[:,2] = dfSegment['z'].to_list()

            # "t","segmentID","segment","radius","modified"
        
            lineString = LineString(zxy)

            # print(segmentID, 'n:', n, 'lineString len:', lineString.length, len(lineString.coords))

            geoDict = {}
            geoDict['t'] = sessionID
            geoDict['segmentID'] = segmentID
            geoDict['segment'] = lineString
            geoDict['radius'] = 4
            geoDict['modified'] = 0

            # dfGeoList.append(geoDict)
            igorDict['dfGeoLineList'].append(geoDict)

    #
    dfGeoLine = pd.DataFrame(igorDict['dfGeoLineList'])
    dfGeoPoints = pd.DataFrame()

    # make a loader with all segments and no spines
    loader = MultiImageLoader(
        lineSegments=dfGeoLine,
        points=dfGeoPoints,
        metadata=igorDict['metadata']
        )

    # append all images to the loader
    logger.info('appending images')
    for _idx, sessionID in enumerate(sessionList):
        if 1 or _idx == 0:
            loader.read(igorDict['imgCh1'][_idx], time=_idx, channel=0)
            loader.read(igorDict['imgCh2'][_idx], time=_idx, channel=1)

    #
    # make map from loader
    logger.info('making map from loader')
    map = MapAnnotations(loader)

    #
    # add all spines to the loader
    import warnings
    warnings.filterwarnings("ignore")

    _totalAdded = 0
    for _idx, sessionID in enumerate(sessionList):
        _t = _idx
        dfPoints = igorDict['dfPoints'][_idx]
        dfSpines = dfPoints[ dfPoints['roiType']=='spineROI' ]
        _count = 0
        for index, row in dfSpines.iterrows():
            # index is the original row before reducing to spineROI
            segmentID = int(row['segmentID'])
            x = row['x']
            y = row['y']
            z = row['z']
            
            # print('ADDING SPINE:', '_count', _count, 'index:', index, 'segmentID:', segmentID, 't:', _t, x, y, z)

            # TODO: add spine does not connect anchor properly
            newSpineID = map.addSpine(segmentId=(segmentID,_t),
                                    x=x, y=y, z=z,
                                    #brightestPathDistance=brightestPathDistance,
                                    #channel=channel,
                                    #zSpread=zSpread
                                    )
            
            _count += 1

        _totalAdded += _count
        print(f'_idx:{_idx} sessionID:{sessionID} add {_count} spines')

    print('total added spine:', _totalAdded)
    
    # save our new map
    if oneTimepoint is None:
        mmMapSessionFile = f'rr30a_tmp2.mmap'
    else:
        mmMapSessionFile = f'rr30a_s{oneTimepoint}_tmp2.mmap'
    savePath = os.path.join('sandbox', mmMapSessionFile)
    logger.info(f'saving: {savePath}')
    map.save(savePath)

    # make sure we can load the map
    print('re-load as map2')
    map2 = MapAnnotations(MMapLoader(savePath).cached())
    logger.info(f'map2:{map2}')

if __name__ == '__main__':
    setLogLevel()
    folder = '../PyMapManager-Data/maps/rr30a'
    importStack(folder)