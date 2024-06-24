import pytest
import mapmanagercore.data

def test_load():
    linesFile = mapmanagercore.data.getLinesFile()
    pointsFile = mapmanagercore.data.getPointsFile()
    ch1 = mapmanagercore.data.getTiffChannel_1()
    ch2 = mapmanagercore.data.getTiffChannel_2()
    mmap = mapmanagercore.data.getSingleTimepointMap()

if __name__ == '__main__':
    test_load()
    
    