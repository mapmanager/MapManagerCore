import time
import numpy as np
import pandas as pd

from mapmanagercore import MapAnnotations
from mapmanagercore.annotations.single_time_point import SingleTimePointAnnotations
from mapmanagercore.logger import logger

def getPlotDict_mpl():
    """Get a new default plot dictionary.

    The plot dictionary is used to tell plot functions what to plot (e.g. ['xstat'] and ['ystat']).
    
    All plot function return the same plot dictionary with keys filled in with values that were plotted
    (e.g. ['x'] and ['y']).
    
    Example::
    
    	import mapmanagercore.data
        from mapmanagercore import MapAnnotations
        from pymapmanager.coreUtils import getPlotDict_mpl, getMapValues3
    	
    	path = mapmanagercore.data.getMultiTimepointMap()
    	map = MapAnnotations.load(self.path)
    	plotdict = getPlotDict_mpl()
    	plotdict['xstat'] = 'days'
    	plotdict['ystat'] = 'pDist' # position of spine on its parent segment
    	plotdict = getMapValues3(mmmap, plotdict)
    	
    	# display with matplotlib
    	plotdict['x']
    	plotdict['y']
    	
    """
    PLOT_DICT = {
        'map' : None, #: map (object) mmMap
        'mapname' : None,
        'sessidx' : None, #: sessIdx (int): map session
        'stack' : None, #: stack (object) use for single timepoint analysis

        'xstat' : None, #: xstat (str): Name of statistic to retreive, corresponds to column in stack.stackdb
        'ystat' : None,
        'zstat' : None,
        'roitype' : ['spineROI'], #: roiType
        'segmentid' : [],

        'stacklist' : [],   # list of int to specify sessions/stacks to plot, [] will plot all

        'getMapDynamics' : True, # set True to get map 'dynamics'
        
        'plotbad' : True,
        'plotintbad' : False,
        'showlines' : True,
        'linewidth': 1,
        'showdynamics': True,
        'markersize' : 15,
        'doDark': True,

        #  Filled in by get functions
        'x' : None,
        'y' : None,
        'z' : None,
        'stackidx' : None,
        'reverse' : None,
        'runrow': None,
        'mapsess': None,
    }
    return PLOT_DICT

def _GuessConnectedSpines(map : MapAnnotations,
                        tp1,
                        tp2,
                        segment1,
                        segment2,
                        thesholdDist : float = 10) -> pd.DataFrame:
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

    def makeNp(tp : SingleTimePointAnnotations, segmentID):
        """abb What is this doing???
        """
        points = tp.points[:]
        
        # logger.error(f'segmentID:{segmentID}')
        
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

    return dfRet

def buildConnectDataframe(mapAnnotations : MapAnnotations, tp1, tp2, segmentID):
    """Build a DataFrame for connecting spines
    """
    
    # full dataframe
    df = mapAnnotations.points[:]
    df = df.reset_index()

    # reduce to tp1, segmentID
    dfTp1 = df[ (df['t']==tp1) & (df['segmentID']==segmentID)]

    columns = ['Pre ID', 'Pre Pos', 'Post ID', 'Post Pos', 'Distance', 'Guess ID']
    dfRet = pd.DataFrame(columns=columns)

    # step 1, make a df with tp1
    dfRet[['Pre ID', 'Pre Pos']] = dfTp1[['spineID', 'spinePosition']]
    
    # used to append tp2 spineID not in tp1 (added spines)
    appendRowList = []

    # step 2, insert values of connected spines from tp2 into tp1
    dfTp2 = df[ (df['t']==tp2) & (df['segmentID']==segmentID)]
    for index, row in dfTp2.iterrows():
        postID = row['spineID']
        postSpinePosition = row['spinePosition']

        # find tp2 row in tp1
        rowTp1 = dfRet.loc[dfRet['Pre ID'] == postID]
        if len(rowTp1) > 1:
            logger.error(f'got more than one row {rowTp1}')
        elif len(rowTp1) == 1:
            # assign
            dfRet.loc[rowTp1.index, 'Post ID'] = postID
            dfRet.loc[rowTp1.index, 'Post Pos'] = postSpinePosition

            preSpinePosition = dfRet.loc[rowTp1.index, 'Pre Pos']
            dist = postSpinePosition - preSpinePosition
            dfRet.loc[rowTp1.index, 'Distance'] = dist

        else:
            # append to end
            appendRowList.append({
                'Post ID': postID,
                'Post Pos': postSpinePosition
            })
    
    # works if appendRowList is []
    dfRet = pd.concat([dfRet, pd.DataFrame(appendRowList)], ignore_index=True, sort=False)

    # round results
    decimals = 2    
    dfRet['Pre Pos'] = dfRet['Pre Pos'].apply(lambda x: round(x, decimals))
    dfRet['Post Pos'] = dfRet['Post Pos'].apply(lambda x: round(x, decimals))
    dfRet['Distance'] = dfRet['Distance'].apply(lambda x: round(x, decimals))

    # fill in post guess

    # print(dfRet)
    return dfRet

def getMapValues3(mapAnnotations : MapAnnotations,  plotDict : dict):
    """Get values of a stack annotation across all stacks in the map.

    Args:
        plotDict (dict):
            A plot dictionary describing what to plot.
            Get default from getPlotDict_mpl().

    Returns:

        | pd['x'], 2D ndarray of xstat values, rows are runs, cols are sessions, nan is where there is no stackdb annotation
        | pd['y'], same
        | pd['z'], same
        | pd['stackidx'], Each [i]j[] gives the stack centric index of annotation value at [i][j].
        | pd['mapsess'], Each [i][j] gives the map session of value at annotation [i][j].
        | pd['runrow'],

    """
    startTime = time.time()

    segmentID = plotDict['segmentid']
    xStat = plotDict['xstat']
    yStat = plotDict['ystat']

    logger.info(f'fetching plot dict for segmentID:{segmentID} xStat:{xStat} yStat:{yStat}')

    df = mapAnnotations.points[:]
    
    # move row labels (spineID, t) to columns
    df = df.reset_index()

    # logger.info('df is:')
    # print(df)

    # reduce to segment id
    df = df[ df['segmentID']==segmentID]

    # scatter
    xPlot = df[xStat].to_list()
    yPlot = df[yStat].to_list()
    
    # each point in scatter has a spine id
    # Note: spineID goes across timepoints
    xyPlotSpineID = df['spineID'].to_list()

    # each spineID has a timepoint
    xyPlotTimepoint = df['t'].to_list()

    # each spine has a segment
    xySegmentID = df['segmentID'].to_list()

    xyAccept = df['accept'].to_list()

    markerColor = ['w'] * len(xyPlotSpineID)
    # lines
    
    #new
    xSpineLineDict = {}
    ySpineLineDict = {}
    spineRunDict = {}

    spineIDs = df['spineID'].unique()  # step through connected spineID
    xPlotLines = []
    yPlotLines = []
    for spineID in spineIDs:
        # spineID = int(spineID)
        
        # grab rows of a spineID (across timepoints)
        spineDf = df[ df['spineID']== spineID]
        
        xPlotLine = spineDf[xStat].to_list()
        yPlotLine = spineDf[yStat].to_list()
        
        # this is redundant, merge with xPlineLineDict
        xPlotLines.append(xPlotLine)
        yPlotLines.append(yPlotLine)
        
        spineID_int = int(spineID)
        xSpineLineDict[spineID_int] = xPlotLine
        ySpineLineDict[spineID_int] = yPlotLine
        
        spineRunDict[spineID_int] = spineDf.index.astype(int)

        # set color dynamics
        xt = spineDf['t'].to_list()
        if len(xt) == 1:
            # transient
            _x1 = np.where((np.array(xyPlotSpineID==spineID)))[0]
            markerColor[_x1[0]] = 'b'
        else:
            if xt[0] != 0:
                # added
                _x1 = np.where((np.array(xyPlotSpineID==spineID)))[0]
                markerColor[_x1[0]] = 'g'
            if xt[-1] != 4:
                # deleted
                _x1 = np.where((np.array(xyPlotSpineID)==spineID) & (np.array(xyPlotTimepoint)==xt[-1]))[0]
                # logger.info(f'spineID:{spineID} at tp {xt[-1]} is SUBTRACTED _finalIndex:{_finalIndex}')
                markerColor[_x1[0]] = 'r'
            
    # return values
    
    # old
    plotDict['x'] = xPlot
    plotDict['y'] = yPlot
    plotDict['xyPlotSpineID'] = xyPlotSpineID
    plotDict['xyPlotTimepoint'] = xyPlotTimepoint
    plotDict['xPlotLines'] = xPlotLines
    plotDict['yPlotLines'] = yPlotLines

    plotDict['markerColor'] = markerColor

    plotDict['xSpineLineDict'] = xSpineLineDict
    plotDict['ySpineLineDict'] = ySpineLineDict
    plotDict['spineRunDict'] = spineRunDict

    # new
    # dfPlot = pandas.DataFrame()
    # dfPlot['x'] = xPlot
    # dfPlot['y'] = yPlot
    # dfPlot['spineID'] = xyPlotSpineID
    # dfPlot['timepoint'] = xyPlotTimepoint
    # dfPlot['segmentID'] = xySegmentID
    # dfPlot['dynamics'] = ''
    # dfPlot['dynamics'] = xyAccept
    
    # pd['dfPlot'] = dfPlot

    stopTime = time.time()
    logger.info(f'   took:{round(stopTime - startTime, 2)} seconds')

    return plotDict

class ConnectSpines:
    """Manage connected spines between two timepoints.
    """
    def __init__(self, map, tp1, tp2, segmentID, threshold=10):
        self._map = map
        self.tp1 = tp1
        self.tp2 = tp2
        self.segmentID = segmentID

        self.df : pd.DataFrame = None  # set in setTimepoints

        #columns = ['Pre ID', 'Pre Pos', 'Post ID', 'Post Pos', 'Distance', 'Guess ID']
        self.setTimepoints(tp1, tp2, segmentID, threshold)

    def setTimepoints(self, tp1, tp2, segmentID, threshold) -> pd.DataFrame:
        self.df = buildConnectDataframe(self._map, tp1, tp2, segmentID)
        self.dfLastGuess = self.guessConnections(threshold)
        # self._addGuess()
        return self.df

    def guessConnections(self, threshold : float) -> pd.DataFrame:
        """Update guess with new threshold.
        """
                
        # df of best guess connected spines
        df = _GuessConnectedSpines(self._map,
                                 self.tp1,
                                 self.tp2,
                                 segment1=self.segmentID,
                                 segment2=self.segmentID,
                                 thesholdDist=threshold)
        self._addGuess(df)
        return self.df
    
    def _addGuess(self, newGuess : pd.DataFrame):
        """Add new guess to main pre/post df.
        """
        logger.error('')
        for index, row in newGuess.iterrows():
            preSpineID = row['spineID']
            preSpineID = int(preSpineID)
            postSpineID = row['toSpineID']  # might be nan
            if np.isnan(postSpineID):
                # don't add
                continue
            postSpineID = int(postSpineID)
            # get preSpineID row
            # preRow = self.df[ self.df['Pre ID']==preSpineID]
            # set Guess ID
            self.df.loc[preSpineID, 'Guess ID'] = postSpineID
                        
if __name__ == '__main__':
    logger.setLevel('DEBUG')
    from mapmanagercore import MapAnnotations
    import mapmanagercore.data
    
    path = mapmanagercore.data.getMultiTimepointMap()

    map : MapAnnotations = MapAnnotations.load(path)

    # disturbing we have to do this !!!!
    # map.points[:]

    tp1 = 1
    tp2 = 2
    segmentID = 0
    # buildConnectDataframe(map, tp1, tp2, segmentID)
    cs = ConnectSpines(map, tp1, tp2, segmentID)
    print(cs.df)

    # cs.connect(3, 278)
    # cs.connect(1, 1)