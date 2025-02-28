# abj
# Testing cases for appendSegmentPoint() along with new brightest-path-tracing

import unittest
from mapmanagercore.logger import logger
from mapmanagercore.annotations.mutation import AnnotationsBaseMut
from mapmanagercore.lazy_geo_pd_images.loader.base import ImageLoader
from mapmanagercore.schemas.spine import Spine
from tests import test_single_timepoint_map

class TestAppendSegmentPoint(unittest.TestCase):
    # def new(self):
    #     return AnnotationsBaseMut(ImageLoader())

    def test_simple_append(self):
        """ Simple append of one segment point
        """
        map = test_single_timepoint_map.test_create_map()
        tp = map.getTimePoint(time=0)
        newSegmentID = tp.newSegment()

        x = 811 
        y = 939 
        z = 32 
        tp.appendSegmentPoint(newSegmentID, x, y, z)
        self.assertEqual(len(tp._segments), 1)
    
    def test_same_first_point_append(self):
        """ Edge Case where user clicks same segment point twice

        Do not add the second point since it is the same
        """
        map = test_single_timepoint_map.test_create_map()
        tp = map.getTimePoint(time=0)
        newSegmentID = tp.newSegment()

        x = 811 
        y = 939 
        z = 32 
        tp.appendSegmentPoint(newSegmentID, x, y, z)
        self.assertEqual(len(tp._segments), 1)
        
        x = 811 
        y = 939 
        z = 32 
        tp.appendSegmentPoint(newSegmentID, x, y, z)
        self.assertEqual(len(tp._segments), 1)   
        
        # logger.info(f"tp._segments {tp._segments[:]}")
        # logger.info(f"len of tp._segments {len(tp._segments[:])}")

    def test_higher_to_lower_append(self):
        """ This tests when a user adds two points. The first point being lower in coordinate value and the second being higher 
        in coordinate value. For brightest-Path-Tracing, the order matters
        """
        logger.info(f"here test")
        map = test_single_timepoint_map.test_create_map(logData = False)
        tp = map.getTimePoint(time=0)
        newSegmentID = tp.newSegment()

        x = 811 
        y = 939 
        z = 32 
        start_point = (x,y,z)
        tp.appendSegmentPoint(newSegmentID, x, y, z)

        x = 729 
        y = 908 
        z = 32 
        end_point = (x,y,z)
        tp.appendSegmentPoint(newSegmentID, x, y, z)

        newSegment = tp._segments[newSegmentID]
        # check that first and last point of segment matches that of the appended points
        unload = newSegment[:]
        geom = unload.loc[1, 'segment']
        segmentPoint0 = geom.coords[0]
        segmentPointLast = geom.coords[-1]
        
        logger.info(f"segmentPoint0 {segmentPoint0}")
        logger.info(f"segmentPointLast {segmentPointLast}")
        # Note: the "start_point" within brightest path tracing is the last added point
        self.assertEqual(segmentPointLast, start_point)
        self.assertEqual(segmentPoint0, end_point)

    def test_lower_to_higher_append(self):
        """ This tests when a user adds two points. The first point being higher in coordinate value and the second being lower 
        in coordinate value. For brightest-Path-Tracing, the order matters
        """
        logger.info(f"here test")
        map = test_single_timepoint_map.test_create_map(logData = False)
        tp = map.getTimePoint(time=0)
        newSegmentID = tp.newSegment()

        x = 729 
        y = 908 
        z = 32 
        start_point = (x,y,z)
        tp.appendSegmentPoint(newSegmentID, x, y, z)

        x = 811 
        y = 939 
        z = 32 
        end_point = (x,y,z)
        tp.appendSegmentPoint(newSegmentID, x, y, z)

        newSegment = tp._segments[newSegmentID]
        # check that first and last point of segment matches that of the appended points
        unload = newSegment[:]
        geom = unload.loc[1, 'segment']
        segmentPoint0 = geom.coords[0]
        segmentPointLast = geom.coords[-1]
        
        # logger.info(f"segmentPoint0 {segmentPoint0}")
        # logger.info(f"segmentPointLast {segmentPointLast}")
        # Note: the "start_point" within brightest path tracing is the last added point
        self.assertEqual(segmentPointLast, start_point)
        self.assertEqual(segmentPoint0, end_point)

    # def 

if __name__ == '__main__':
    unittest.main()
