import pandas as pd

from mapmanagercore import MapAnnotations, MultiImageLoader
from mapmanagercore.logger import logger
import mapmanagercore.data

def test_create_map() -> MapAnnotations:
    """Create a single tp, one channel map.
    """
    ch1_path = mapmanagercore.data.getTiffChannel_1()
    ch2_path = mapmanagercore.data.getTiffChannel_2()

    loader = MultiImageLoader()

    tp = 0
    loader.read(ch1_path, channel=0, time=tp)
    loader.read(ch2_path, channel=1, time=tp)

    # abb I would like this to accept None for lineSegments and points
    map = MapAnnotations(loader,
                        #  lineSegments=gp.GeoDataFrame(),
                        #  points = gp.GeoDataFrame())
                         lineSegments=pd.DataFrame(),
                         points = pd.DataFrame())
    return map

def test_add_segment():
    map = test_create_map()

    # would be nice if we could create a new segment by specifying the timepoint
    tp = map.getTimePoint(time=0)
    
    newSegmentID = tp.newSegment()

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
    # map = test_create_map()
    # print(map)

    test_add_segment()
