import pandas as pd
import pytest
import mapmanagercore.data

def test_load():
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


if __name__ == '__main__':
    test_load()

    