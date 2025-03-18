from pprint import pprint
import pandas as pd

from mapmanagercore import MapAnnotations, MultiImageLoader
from mapmanagercore.logger import logger
import mapmanagercore.data

def test_create_map(logData: bool = True) -> MapAnnotations:
    """Create a single tp, one channel map.
    """
    ch1_path = mapmanagercore.data.getTiffChannel_1()
    ch2_path = mapmanagercore.data.getTiffChannel_2()

    # abb old loader
    loader = MultiImageLoader()

    tp = 1  # abb 20250317 timepoints are 1 based
    loader.read(ch1_path, time=tp, channel=0)
    loader.read(ch2_path, time=tp, channel=1)

    # check metadata
    md0 = loader.metadata(t=1)

    if logData:
        logger.info('metadata for loader t=0 is:')
        pprint(md0)

        logger.info('load._metadata3 is:')
        pprint(loader._metadata3)

    # md1 = loader.metadata(t=1)
    # logger.info('metadata for channel 1 is:')
    # pprint(md1)

    # abb I would like this to accept None for lineSegments and points
    map = MapAnnotations(loader,
                         lineSegments=pd.DataFrame(),
                         points = pd.DataFrame())
    return map

def test_add_segment():
    map = test_create_map()

    # would be nice if we could create a new segment by specifying the timepoint
    # timepoints are 1 based
    # tp = map.getTimePoint(time=0)
    timepoint = 1  # timepoints are 1 based
    tp = map.getTimePoint(time=timepoint)
    
    newSegmentID = tp.newSegment()
    logger.info(f'newSegmentID:{newSegmentID}')
    
    # fails because the segment has no points
    # tp.addSpine(newSegmentID, x=20, y=30, z=12)

    # add some points to the segment
    x = 100
    y = 200
    z = 12
    tp.appendSegmentPoint(newSegmentID, x, y, z)

    # TODO add check on addSpine() and
    # do not add if segment has 1 point
    # tp.addSpine(newSegmentID, x=20, y=30, z=12)

    # add some points to the segment
    x = 110
    y = 210
    z = 14
    tp.appendSegmentPoint(newSegmentID, x, y, z)

    tp = map.getTimePoint(time=0)
    tp.addSpine(newSegmentID, x=20, y=30, z=12)

    print(map)

if __name__ == '__main__':
    logger.setLevel('DEBUG')
    # mapAnnotations = test_create_map()
    # print(mapAnnotations)

    test_add_segment()
