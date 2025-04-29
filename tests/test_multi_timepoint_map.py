# import pytest
from pprint import pprint

import mapmanagercore.data
from mapmanagercore import MapAnnotations
from mapmanagercore.logger import logger

def _old_test_multi_timepoint():

    mmapPath = mapmanagercore.data.getMultiTimepointMap()
    print(f'mmapPath:{mmapPath}')

    # check we can load a map
    # ok = MapAnnotations.checkFile(mmapPath, verbose=False)
    # print(f'ok:{ok}')
    # assert ok

    # actually load the map
    map = MapAnnotations.load(mmapPath)
    tp = map.getNumTimepoints()
    assert tp == 5

    deleted = map.loader.deleteTimePoint(1)
    print(deleted, map.getNumTimepoints())
    
    timePoints = map.loader.timePoints()
    print(f'after delete timePoints:{timePoints}')

    # tp channels
    # for tp in map.loader.timePoints():
    #     print(f'tp:{tp} channels:{map.loader.channels(tp)}')
    #     md = map.loader.metadata(tp)
    #     pprint(md)

    logger.info('=== TESTING NEW metadata3')
    
    # append channel from tiff
    tiffPath = mapmanagercore.data.getTiffChannel_1()
    map.loader.appendSingleTimepointTif(tiffPath)  # abb md3
    timePoints = map.loader.timePoints()
    lastTimepoint = timePoints[-1]
    # print(f'after appendSingleTimepointTif() timePoints:{timePoints} lastTimepoint:{lastTimepoint}')
    # md = map.loader.metadata(lastTimepoint)
    # pprint(md)

    # append tiff to timepoint
    tiffPath2 = mapmanagercore.data.getTiffChannel_2()
    map.loader.appendTiffToTimepoint(lastTimepoint, tiffPath2)  # abb md3
    # print('after append second channel')
    # md = map.loader.metadata(lastTimepoint)
    # pprint(md)

    # check metadata3
    print(f'md3 has numtimepoint: {map.loader._metadata3.numTimepoints}')
    print(f'lastTimepoint:{lastTimepoint}')
    pprint(map.loader._metadata3[lastTimepoint])

if __name__ == '__main__':
    test_multi_timepoint()