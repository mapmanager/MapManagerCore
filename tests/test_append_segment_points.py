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

    # 9 Test Cases for Brightest Path
    # Currently Brightest Path uses a reduced image for speeding up processing time
    # This reduced image defines a new lower and upper bound of xMin,yMin and xMax,yMax
    # Since brightestPath() calls brightest_path_lib.algorithm.AStarSearch
    # with the reduced image and points, 
    # this creates different cases for the comparison of the two added points
    # Ex: With Point 1 being (x1,x2) and Point 2 being (x2,y2), x1 < x2 and y1 < y2

    def pointComparison(self, point1: tuple, point2: tuple):
        """

        point1: first point added in form of (x,y,z)
        point2: second point added in form of (x,y,z)
        """
        map = test_single_timepoint_map.test_create_map(logData = False)
        tp = map.getTimePoint(time=0)
        newSegmentID = tp.newSegment()

        x,y,z = point1
        start_point = (x,y,z)
        tp.appendSegmentPoint(newSegmentID, x, y, z)

        x,y,z = point2
        end_point = (x,y,z)
        tp.appendSegmentPoint(newSegmentID, x, y, z)

        newSegment = tp._segments[newSegmentID]
        # check that first and last point of segment matches that of the appended points
        unload = newSegment[:]
        logger.info(f"unloaded segment {unload}")
        geom = unload.loc[1, 'segment']
        segmentPoint0 = geom.coords[0]
        segmentPointLast = geom.coords[-1]
        
        logger.info(f"segmentPoint0 {segmentPoint0}")
        logger.info(f"segmentPointLast {segmentPointLast}")

        # if not isReversed:
        #     self.assertEqual(segmentPoint0, start_point)
        #     self.assertEqual(segmentPointLast, end_point)
        # else:
        self.assertEqual(segmentPointLast, start_point)
        self.assertEqual(segmentPoint0, end_point)

    def test_case_1(self):
        """ This tests when a user adds two points, where x1 < x2 and y1 < y2

        With Point 1 being (x1,x2) and Point 2 being (x2,y2)

        """
        point1 = (811, 939, 32) 
        point2 = (729, 908, 32) 
        self.pointComparison(point1, point2)

    def test_case_2(self):
        """ This tests when a user adds two points, where x1 > x2 and y1 > y2

        With Point 1 being (x1,x2) and Point 2 being (x2,y2)
        """
        point1 = (729, 908, 32) 
        point2 = (811, 939, 32)  
        self.pointComparison(point1, point2)

    def test_case_3(self):
        """ This tests when a user adds two points, where x1 > x2 and y1 < y2:

        With Point 1 being (x1,x2) and Point 2 being (x2,y2)
        """
        point1 = (209, 168, 11) 
        point2 = (250, 143, 11)  
        self.pointComparison(point1, point2)
        # self.pointComparison(point1, point2, isReversed = True)

    def test_case_4(self):
        """ This tests when a user adds two points, where x1 < x2 and y1 > y2:

        With Point 1 being (x1,x2) and Point 2 being (x2,y2)
        """
        point1 = (250, 143, 11)
        point2 = (209, 168, 11)
        self.pointComparison(point1, point2)

    def test_case_5(self):
        """ This tests when a user adds two points, where x1 < x2 and y1 == y2:

        With Point 1 being (x1,x2) and Point 2 being (x2,y2)
        """
        point1 = (593, 401, 20)
        point2 = (584, 401, 20)
        self.pointComparison(point1, point2)

    def test_case_6(self):
        """ This tests when a user adds two points, where x1 > x2 and y1 == y2:

        With Point 1 being (x1,x2) and Point 2 being (x2,y2)
        """
        point1 = (584, 401, 20)
        point2 = (593, 401, 20)
        self.pointComparison(point1, point2)

    def test_case_7(self):
        """ This tests when a user adds two points, where x1 == x2 and y1 > y2:

        With Point 1 being (x1,x2) and Point 2 being (x2,y2)
        """
        point1 = (237, 614, 25)
        point2 = (237, 631, 25)
        self.pointComparison(point1, point2)

    def test_case_8(self):
        """ This tests when a user adds two points, where x1 == x2 and y1 < y2:

        With Point 1 being (x1,x2) and Point 2 being (x2,y2)
        """
        point1 = (237, 631, 25)
        point2 = (237, 614, 25)
        self.pointComparison(point1, point2)

    def test_case_9(self):
        """ This tests when a user adds two points, where x1 == x2 and y1 == y2:

        With Point 1 being (x1,x2) and Point 2 being (x2,y2)
        """
        
        # When points are equal it is the same as this previous test case:
        self.test_same_first_point_append()

if __name__ == '__main__':
    unittest.main()
