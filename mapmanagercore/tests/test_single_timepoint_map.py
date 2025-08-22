from pprint import pprint
import pandas as pd

from mapmanagercore.lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader
from mapmanagercore import MapAnnotations
from mapmanagercore.logger import logger
import mapmanagercore.data

def test_create_map() -> MapAnnotations:
    """Create a single tp, one channel map.
    """
    ch1_path = mapmanagercore.data.getTiffChannel_1()
    ch2_path = mapmanagercore.data.getTiffChannel_2()

    # abb old loader
    loader = mmMapLoader()
    timepoint_key = loader.importTimepoint(ch1_path)  # abai 20250806

    # check metadata
    md0 = loader.getTimepointMetadata(t=1)

    # if logData:
    #     logger.info('metadata for loader t=1 is:')
    #     pprint(md0)

    #     logger.info('load._metadata3 is:')
    #     pprint(loader.metadata)

    # md1 = loader.metadata(t=1)
    # logger.info('metadata for channel 1 is:')
    # pprint(md1)

    # abb I would like this to accept None for lineSegments and points
    map = MapAnnotations(loader,
                         lineSegments=pd.DataFrame(),
                         points = pd.DataFrame())
    
    logger.info(f'import 2nd channel timepoint_key:{timepoint_key}')
    map.loader.importChannel(ch2_path, timepoint_key)

    return map

def test_add_segment():
    map = test_create_map()

    # would be nice if we could create a new segment by specifying the timepoint
    # timepoints are 1 based
    # tp = map.getTimePoint(time=0)
    timepoint = 1  # timepoints are 1 based
    tp = map.getTimePoint(time=timepoint)

    logger.info('empty tp:')
    logger.info(f'  {tp}')

    newSegmentID = tp.newSegment()
    logger.info(f'newSegmentID:{newSegmentID}')
    logger.info('after add segment tp:')
    logger.info(f'  {tp}')


    # fails because the segment has no points
    # tp.addSpine(newSegmentID, x=20, y=30, z=12)

    # add some points to the segment
    segmentPoints = [
        (100, 200, 12),
        (110, 210, 14),
    ]
    _pntIdx = 0
    for x, y, z in segmentPoints:
        tp.appendSegmentPoint(newSegmentID, x, y, z)
        logger.info(f'after appendSegmentPoint {_pntIdx}, tp is:')
        logger.info(f'  {tp}')
        _pntIdx += 1

    spinePoints = [
        (20, 30, 12),
        (30, 40, 15),
    ]
    _spineIdx = 0
    for x, y, z in spinePoints:
        tp.addSpine(newSegmentID, x=x, y=y, z=z)
        logger.info(f'after addSpine {_spineIdx}, tp is:')
        logger.info(f'  {tp}')
        _spineIdx += 1

    # logger.info('after addSpine tp:')
    # logger.info(f'  {tp}')

    # we added to a timepoint, check the map
    logger.info('original map is:')
    logger.info(f'  {map}')

    print(map.points)  # LazyGeoFrame


    # test deleteSpines
    
    # delete using map
    # spineId = (1, 1)  # (spineID, timepoint)
    # map.deleteSpine(spineId)

    # delete using tp.deleteSpine
    deleteSpineID = 1
    timepoint = 1
    tp.deleteSpine(deleteSpineID, timepoint)

    logger.info('after deleteSpine, tp is:')
    logger.info(f'  {tp}')
    logger.info('after deleteSpine, map is:')
    logger.info(f'  {map}')

    # test deleteSegment (will fail if there are spines)
    # segmentId = (newSegmentID, 1)  # 1 based index
    # forceDelete = False
    # map.deleteSegment(segmentId, forceDelete=forceDelete)
    # logger.info('after deleteSegment, map is:')
    # logger.info(f'  {map}')

if __name__ == '__main__':
    logger.setLevel('DEBUG')
    # mapAnnotations = test_create_map()
    # print(mapAnnotations)

    test_add_segment()
