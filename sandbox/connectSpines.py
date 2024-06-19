import numpy as np
import pandas as pd

from mapmanagercore import MapAnnotations, MMapLoader
from mapmanagercore.annotations.single_time_point import SingleTimePointAnnotations

from mapmanagercore.logger import logger

def connectSpines(map :MapAnnotations, tp1, tp2, segment1, segment2, thesholdDist : float = 10):
    """Connect spines between tp1 and tp2.

    Parameters
    ---------
    tp1,tp2 : int
        The timepoint (e.g. time or t) to connect between


    Returns
    -------
    pd.DataFrame with columns
        spineID
        position
        toSpineID
        toPosition
        dist
    """
    spineID = 0
    position = 1
    isLeft = 2
    toSpineID = 3
    toDistance = 4
    toPosition = 5
    # spineLength = 6
    # toSpineLength = 7

    # thesholdDist = 10
    # print('thesholdDist:', thesholdDist)

    def makeNp(tp : SingleTimePointAnnotations, segmentID):  # , isFrom : bool):
        points = tp.points[:]
        points = points[points['segmentID']==segmentID]

        points['isLeft'] = (points['spineSide']=='Left')

        # print(points['spineLength'].max())  # about 25

        m = len(points)

        _np = np.zeros(shape=(m,6))
        _np[:,spineID] = points.index.to_list()
        _np[:,position] = points['spinePosition']
        _np[:,isLeft] = points['isLeft']  # left -> 1, right -> 0

        _np[:,toSpineID] = np.nan
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
        fromNp[i,toDistance] = dist
        fromNp[i, toPosition] = toNp[j,position]
        
        toNp[j, toSpineID] = fromNp[i, spineID]  # int(i)
        toNp[j, toDistance] = dist
        toNp[j, toPosition] = fromNp[i,position]  # not used

    def disconnect(i, j):
        fromNp[i,toSpineID] = np.nan
        fromNp[i,toDistance] = np.nan
        fromNp[i,toPosition] = np.nan
        
        toNp[j, toSpineID] = np.nan
        toNp[j, toDistance] = np.nan
        toNp[j, toPosition] = np.nan

    _fromTp : SingleTimePointAnnotations = map.getTimePoint(tp1)
    _toTp : SingleTimePointAnnotations = map.getTimePoint(tp2)

    fromNp = makeNp(_fromTp, segment1)
    toNp = makeNp(_toTp, segment2)

    m = len(fromNp)
    n = len(toNp)
    
    numTieBreakers = -1
    numIteration = 0
    while numTieBreakers != 0:
        logger.info(f'iteration:{numIteration} tie breakers:{numTieBreakers}')

        numTieBreakers = 0
        for i in range(m):
            iLeft = fromNp[i,isLeft]
            iIsTaken = ~np.isnan(fromNp[i, toSpineID])
            for j in range(n):
                jLeft = toNp[j,isLeft]
                if iLeft != jLeft:
                    continue
                
                jIsTaken = ~np.isnan(toNp[j, toSpineID])
                dist = abs(fromNp[i,position] - toNp[j,position])

                if dist < thesholdDist:
                    if jIsTaken:
                        existingDist = toNp[j,toDistance]
                        if dist < existingDist:
                            numTieBreakers += 1
                            _toSpineID = int(toNp[j,toSpineID])
                            disconnect(_toSpineID, j)
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

    logger.info(f'numIteration:{numIteration}')

    dfRet = pd.DataFrame()
    dfRet['spineID'] = fromNp[:, spineID]
    dfRet['position'] = fromNp[:, position]
    
    dfRet['toSpineID'] = fromNp[:, toSpineID]
    dfRet['toPosition'] = fromNp[:, toPosition]

    dfRet['dist'] = dfRet['toPosition'] - dfRet['position']

    dfRet['spineLength'] = _fromTp.points[:].loc[dfRet['spineID']]['spineLength']

    # toSpineID will have nan
    _df = dfRet['toSpineID'].dropna().to_list()
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

def debugConnectSpines(map :MapAnnotations):
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

def debugMultiIndex():

    _index = [
        (0,0),
        (2,0),
        
        (2,1),
        (5,1)
    ]
    df = pd.DataFrame(index=_index)
    df['a'] = [
        'a', 'b', 'c', 'd']
    print(df)

    _indexSlice = pd.IndexSlice[2,:]  # all spines with id 5
    print('xxx:')
    print(df.loc[_indexSlice])

def _plotNp(dfRet):
    fromPosition = dfRet['position']
    toPosition = dfRet['toPosition']
    
    xPlot = []
    yPlot = []
    for idx, position in enumerate(fromPosition):
        yPlot.append(position)
        yPlot.append(toPosition[idx])
        yPlot.append(np.nan)

        xPlot.append(0)
        xPlot.append(1)
        xPlot.append(np.nan)

    import matplotlib.pyplot as plt
    plt.plot(xPlot, yPlot, 'o-')
    plt.show()

if __name__ == '__main__':
    
    # a 2 session map
    path = 'sandbox/data/rr30a_2tp.mmap'
    map = MapAnnotations(MMapLoader('/Users/cudmore/Sites/MapManagerCore/sandbox/data/rr30a_2tp.mmap'))

    tp1 = 0
    tp2 = 1
    segment1 = 0
    segment2 = 0
    thesholdDist = 10
    df = connectSpines(map, tp1, tp2, segment1, segment2, thesholdDist=thesholdDist)

    # debugConnectSpines(map)

    # debugMultiIndex()

    # a plot for debugging
    _plotNp(df)