import pandas as pd
import pytest

import mapmanagercore.data

def test_load():
    """Check that we can load files from mapmanagercore.data
    
    This loads from a different repo mapmanagercore-data
    """
    
    linesFile = mapmanagercore.data.getLinesFile()
    
    dfLines = pd.read_csv(linesFile)
    print('=== test_load dfLines')
    print(dfLines)

    pointsFile = mapmanagercore.data.getPointsFile()
    dfPoints = pd.read_csv(pointsFile)
    print('=== test_load dfPoints')
    print(dfPoints)

    ch1 = mapmanagercore.data.getTiffChannel_1()

    ch2 = mapmanagercore.data.getTiffChannel_2()

    mmap = mapmanagercore.data.getSingleTimepointMap()

def test_check_file():
    from mapmanagercore import MapAnnotations

    mmapPath = mapmanagercore.data.getSingleTimepointMap()
    
    # check we can load a map
    ok = MapAnnotations.checkFile(mmapPath, verbose=False)
    assert ok

    # actually load the map
    map = MapAnnotations.load(mmapPath)
    assert map is not None

if __name__ == '__main__':
    # test_load()
    test_check_file()
    