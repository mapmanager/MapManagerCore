import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mapmanagercore import MapAnnotations  # , MMapLoader
from mapmanagercore.annotations.single_time_point import SingleTimePointAnnotations

from mapmanagercore.logger import logger

def _getConnectedSpines(map : MapAnnotations, tp1, tp2, segment1, segment2, thesholdDist : float = 10):
    """Get connect spines between tp1 and tp2.

    Parameters
    ---------
    tp1,tp2 : int
        The timepoint (e.g. time or t) to connect between
    segment1, segment2 : int
        The segment ID
        Note: segment ID needs to be the same
    thesholdDist : int
        The threshold to connect spines.
        If <thesholdDist then connect, otherwise do not

    Returns
    -------
    pd.DataFrame with columns
        spineID
        position
        toSpineID
        toPosition
        dist
    """
    
    # second dimension (column index) into our internal 2D numpy array
    spineID = 0
    # segmentID = 1
    position = 2
    isLeft = 3
    toSpineID = 4
    # toSegmentID = 5
    toDistance = 6
    toPosition = 7
    
    _totalNumColumns = 8

    def makeNp(tp : SingleTimePointAnnotations, segmentID):  # , isFrom : bool):
        """What is this doing???
        """
        points = tp.points[:]
        points = points[points['segmentID']==segmentID]

        points['isLeft'] = (points['spineSide']=='Left')

        # print(points['spineLength'].max())  # about 25

        # columns = ['spineID', 'position', 'isLeft', 'toSpineID', 'toPosition', 'toDistance']
        # df = pd.DataFrame(columns=columns)
        # df['spineID'] = points.index.to_list()
        # df['position'] = points['spinePosition']
        # df['isLeft'] = (points['spineSide']=='Left')
        # df['toSpineID'] = np.nan
        # df['toPosition'] = np.nan
        # df['toDistance'] = np.nan
        
        m = len(points)

        _np = np.zeros(shape=(m,_totalNumColumns))
        _np[:,spineID] = points.index.to_list()
        # _np[:,segmentID] = points['segmentID']
        _np[:,position] = points['spinePosition']
        _np[:,isLeft] = points['isLeft']  # left -> 1, right -> 0

        _np[:,toSpineID] = np.nan
        # _np[:,toSegmentID] = np.nan
        _np[:,toDistance] = np.nan
        _np[:,toPosition] = np.nan

        # _np[:,spineLength] = points['spineLength']
        # _np[:,toSpineLength] = np.nan

        # print('xxx')
        # print(_np[:,spineID])

        return _np
    
    def connect(i, j):
        dist = abs(fromNp[i,position] - toNp[j,position])
        fromNp[i,toSpineID] = toNp[j, spineID]  # int(j)
        # fromNp[i,toSegmentID] = toNp[j, segmentID]  # int(j)
        fromNp[i,toDistance] = dist
        fromNp[i, toPosition] = toNp[j,position]
        
        toNp[j, toSpineID] = fromNp[i, spineID]  # int(i)
        # toNp[j, toSegmentID] = fromNp[i, segmentID]  # int(i)
        toNp[j, toDistance] = dist
        toNp[j, toPosition] = fromNp[i,position]  # not used

    def disconnect(i, j):
        fromNp[i,toSpineID] = np.nan
        # fromNp[i,toSegmentID] = np.nan
        fromNp[i,toDistance] = np.nan
        fromNp[i,toPosition] = np.nan
        
        toNp[j, toSpineID] = np.nan
        # toNp[j, toSegmentID] = np.nan
        toNp[j, toDistance] = np.nan
        toNp[j, toPosition] = np.nan

    _fromTp : SingleTimePointAnnotations = map.getTimePoint(tp1)
    _toTp : SingleTimePointAnnotations = map.getTimePoint(tp2)

    fromNp = makeNp(_fromTp, segment1)
    toNp = makeNp(_toTp, segment2)

    m = len(fromNp)
    n = len(toNp)
    
    # logger.info(f'm:{m} n:{n}')

    numTieBreakers = -1
    numIteration = 0
    while numTieBreakers != 0:
        # logger.info(f'iteration:{numIteration} tie breakers:{numTieBreakers}')

        numTieBreakers = 0
        for i in range(m):
            iLeft = fromNp[i,isLeft]
            iIsTaken = ~np.isnan(fromNp[i, toSpineID])
            for j in range(n):
                jLeft = toNp[j,isLeft]
                if iLeft != jLeft:
                    # spines on opposite left/right sides are never connected
                    continue
                
                jIsTaken = ~np.isnan(toNp[j, toSpineID])
                dist = abs(fromNp[i,position] - toNp[j,position])

                if dist < thesholdDist:
                    if jIsTaken:
                        existingDist = toNp[j,toDistance]
                        if dist < existingDist:
                            numTieBreakers += 1
                            _toSpineID = int(toNp[j,toSpineID])
                            # abb
                            # disconnect(_toSpineID, j)
                            disconnect(i, j)
                            # print(f'jIsTaken broke tie {i} {j} existingDist:{existingDist} new dist:{dist}')
                        else:
                            # j is taken but current (i,j) dist does not beat it
                            continue

                    elif iIsTaken:
                        # check if we are closer, e.g. break a tie
                        existingDist = fromNp[i,toDistance]
                        if dist < existingDist:
                            numTieBreakers += 1
                            disconnect(i, j)
                            # print(f'iIsTaken broke tie {i} {j} existingDist:{existingDist} new dist:{dist}')
                        else:
                            # i is taken but current (i,j) dist does not beat it
                            continue

                    connect(i, j)
                    iIsTaken = True
        #
        numIteration += 1

    # logger.info(f'numIteration:{numIteration}')

    dfRet = pd.DataFrame()
    dfRet['spineID'] = fromNp[:, spineID]  # will be float
    dfRet['timepoint'] = tp1
    dfRet['segmentID'] = segment1  # from segmentID
    dfRet['position'] = fromNp[:, position]
    
    dfRet['toSpineID'] = fromNp[:, toSpineID]  # will be float
    dfRet['toTimepoint'] = tp2
    dfRet['toSegmentID'] = segment2  # to segmentID
    dfRet['toPosition'] = fromNp[:, toPosition]

    dfRet['dist'] = dfRet['toPosition'] - dfRet['position']

    dfRet['spineLength'] = _fromTp.points[:].loc[dfRet['spineID']]['spineLength']

    # toSpineID will have nan
    # _df = dfRet['toSpineID'].dropna().to_list()
    # print(_df)
    # _tmp = _toTp.points[:]['spineLength'].loc[_df].to_list()
    # print(_tmp)

    # dfRet['toSpineLength'] = _toTp.points[:]['spineLength'].loc[_df]
    # print(dfRet['toSpineLength'])

    # print('dfRet:', len(dfRet))

    # print('connecting spines')
    # for idx, row in dfRet.iterrows():
    #     fromSpineID = row['spineID']
    #     toSpineID = row['toSpineID']
    #     if ~np.isnan(fromSpineID) and ~np.isnan(toSpineID):
    #         fromSpineID = int(fromSpineID)
    #         toSpineID = int(toSpineID)
    #         # print(f'connect spine {fromSpineID} to {toSpineID}')
    #         map.connect((fromSpineID, tp1), (toSpineID, tp2))

    return dfRet

def _debugConnectSpines(map :MapAnnotations):
    _tmp = map.points[ map.points['segmentID'] == 0 ]
    
    _indexSlice = pd.IndexSlice[5,:]  # all spines with id 5
    # _indexSlice = pd.IndexSlice[:,0]  # does not work
    # print(_tmp[_indexSlice])

    #print(_tmp.index)

    # tp1 starts with spine id 139
    # tp = map.getTimePoint(1)
    
    # df1.rename(index={1: 'a'})
    map.points.index.rename({(140,1): (2,1)}, inplace=True)

    _indexSlice = pd.IndexSlice[140,:]  # all spines with id 5
    print('_indexSlice:', _indexSlice)
    print(map.points[_indexSlice])

    _indexSlice = pd.IndexSlice[2,:]  # all spines with id 5
    print('_indexSlice:', _indexSlice)
    print(map.points[_indexSlice])

def _plotNp(dfRet):
    """Plot a dendrogram of spine position from tp to tp+1
    """
    fromPosition = dfRet['position']
    toPosition = dfRet['toPosition']
    
    timepoint = dfRet.loc[0, 'timepoint']
    toTimepoint = dfRet.loc[0, 'toTimepoint']

    xPlot = []
    yPlot = []
    for idx, position in enumerate(fromPosition):
        yPlot.append(position)
        yPlot.append(toPosition[idx])
        yPlot.append(np.nan)

        xPlot.append(timepoint)
        xPlot.append(toTimepoint)
        
        xPlot.append(np.nan)

    import matplotlib.pyplot as plt
    plt.plot(xPlot, yPlot, 'o-')
    plt.show()

def actuallyConnectSpines(map : MapAnnotations, thesholdDist= 10):
    """Actually connect spines in map.
    """
    numSeg = 5
    numTimepoints = map.getNumTimepoints()
    n = range(numTimepoints-1)
    
    # debug
    # n = range(2)
    
    for tp1 in n:
        
        for seg in range(numSeg):
            logger.info(f'=== tp {tp1} -> {tp1+1} seg:{seg}')

            df = _getConnectedSpines(map, tp1, tp1+1, seg, seg, thesholdDist=thesholdDist)

            for rowLabel, rowDict in df.iterrows():
                fromSpineID = rowDict['spineID']
                toSpineID = rowDict['toSpineID']
                
                if np.isnan(toSpineID):
                    # nothing to connect to
                    continue

                fromSpineID = int(fromSpineID)
                toSpineID = int(toSpineID)
                
                fromSpine = (fromSpineID, tp1)
                toSpine = (toSpineID, tp1+1)
                
                # logger.info(f'connecting {fromSpine} to {toSpine}')

                connected = map.connect(fromSpine, toSpine)

                # if not connected:
                #     print(df)
                #     break

    map.points[:]
    map.segments[:]
    
    # print('====== DONE map.points[:] is')
    # print(map.getPointDataFrame(t=4))
    savePath = '/Users/cudmore/Desktop/multi_timepoint_map_seg_spine_connected.mmap'
    logger.info(f'saving: {savePath}')
    map.save(savePath)

def getPlotMapDict(map, xStat, yStat, segmentID) -> dict:
    """Get a dict to plot a map.
    """
    startSec = time.time()

    df = map.points[:]

    # move row labels (spineID, t) to columns)
    df = df.reset_index()

    # reduce to segment id
    df = df[ df['segmentID']==segmentID]

    # scatter
    xPlot = df[xStat].to_list()
    yPlot = df[yStat].to_list()
    
    # for an x/y plot, each point in scatter has a spine id
    # Note: spineID goes across timepoints
    xyPlotSpineID = df['spineID'].to_list()

    # each spineID has a timepoint
    xyPlotTimepoint = df['t'].to_list()

    # colorize the scatter of spine (points) to (add, subtract, persistent)
    xPlotColor = np.zeros(len(xPlot))

    # abb to colorize
    # TODO: add
    # list.index[value] as check for ValueError

    # lines
    spineIDs = df['spineID'].unique()
    xPlotLines = []
    yPlotLines = []
    for spineID in spineIDs:
        # grab rows of a spineID (across timepoints)
        spineDf = df[ df['spineID']== spineID]
        
        xPlotLine = spineDf[xStat].to_list()
        yPlotLine = spineDf[yStat].to_list()
        
        xPlotLines.append(xPlotLine)
        yPlotLines.append(yPlotLine)
        
        xt = spineDf['t'].to_list()
        if len(xt) == 1:
            # 
            # logger.info(f'spineID:{spineID} at tp {xt} is transient')
            pass
        else:
            if xt[0] != 0:
                # logger.info(f'spineID:{spineID} at tp {xt} is added at tp {xt[0]}')
                pass
            if xt[-1] != 4:
                # logger.info(f'spineID:{spineID} at tp {xt} is subtracted at tp {xt[-1]}')
                pass

    retDict = {
        'xStat' : xStat,
        'yStat' : yStat,
        
        'xPlot' : xPlot,  # list of values for x-axis plot
        'yPlot' : yPlot,

        'xyPlotSpineID' : xyPlotSpineID,
        'xyPlotTimepoint' : xyPlotTimepoint,

        'xPlotLines' : xPlotLines,
        'yPlotLines' : yPlotLines,
        # 'spine_t' : xt
        
    }

    stopSec = time.time()
    logger.info(f'took {stopSec-startSec} s')

    return retDict

def plotMap(map : MapAnnotations):

    xStat = 't'
    yStat = 'spinePosition'
    segmentID = 0
    plotDict = getPlotMapDict(map, xStat=xStat, yStat=yStat, segmentID=segmentID)
    
    # print('=== xyPlotSpineID')
    # print(plotDict['xyPlotSpineID'])

    # print('=== xyPlotTimepoint')
    # print(plotDict['xyPlotTimepoint'])

    startSec = time.time()

    plt.plot(plotDict['xPlot'], plotDict['yPlot'], 'ok')
    
    #lines
    for idx, xPlotLine in enumerate(plotDict['xPlotLines']):
        yPlotLine = plotDict['yPlotLines'][idx]
        plt.plot(xPlotLine, yPlotLine, '-k')

    # lines
    # spineIDs = df['spineID'].unique()
    # for spineID in spineIDs:
    #     spineDf = df[ df['spineID']== spineID]
        
    #     xt = spineDf['t'].to_list()

    #     xPlot = spineDf[xStat].to_list()
    #     yPlot = spineDf[yStat].to_list()
        
    #     plt.plot(xPlot, yPlot, '-')

    #     if len(xt) == 1:
    #         # logger.info(f'{spineID} is transient')
    #         pass
    #     else:
    #         if xt[0] != 0:
    #             # logger.info(f'{spineID} is added at tp {xt[0]}')
    #             pass
    #         if xt[-1] != 4:
    #             # logger.info(f'{spineID} is subtracted at tp {xt[-1]}')
    #             pass

    stopSec = time.time()
    logger.info(f'took {stopSec-startSec} s')

    plt.show()

def _testSaveLoad(map):
    # Save metadata as Parquet and bytes
    df_buffer = io.BytesIO()
    metadata.to_parquet(df_buffer)
    root.create_dataset("metadata_parquet", data=[df_buffer.getbuffer().tobytes()], dtype=bytes)

    # Read from Parquet
    df_buffer = io.BytesIO()
    df_buffer.write(root["metadata_parquet"][0])
    metadata = pd.read_parquet(df_buffer)
    assert isinstance(metadata, pd.DataFrame)

if __name__ == '__main__':
    
    # a 2 session map
    # path = 'sandbox/data/rr30a_2tp.mmap'
    # path = '/Users/cudmore/Sites/MapManagerCore/sandbox/data/rr30a_2tp.mmap'
    # path = '/Users/cudmore/Sites/MapManagerCore/data/two_timepoint.mmap'
    
    # a map with connected segments and each segments have (disconnected) spines
    path = '/Users/cudmore/Desktop/multi_timepoint_map_seg_connected.mmap'
    
    map = MapAnnotations.load(path)

    print(map)

    if 0:
        tp1 = 0  # 1
        tp2 = 1  # 2
        segment1 = 1
        segment2 = 1
        thesholdDist = 10
        df = _getConnectedSpines(map, tp1, tp2, segment1, segment2, thesholdDist=thesholdDist)

        # _debugConnectSpines(map)
        # _debugMultiIndex()

        # a plot for debugging
        _plotNp(df)

        print(df)

    if 0:
        actuallyConnectSpines(map)

    # reload results of actuallyConnectSpines()
    savePath = '/Users/cudmore/Desktop/multi_timepoint_map_seg_spine_connected.mmap'
    logger.info(f'loading map:{savePath}')
    map = MapAnnotations.load(savePath)
    
    print(map)
    print(map.points[:])

    plotMap(map)
