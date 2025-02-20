from typing import Union

import brightest_path_lib.algorithm
from mapmanagercore.utils import injectLine
from .base import SingleTimePointAnnotationsBase
from shapely.geometry import LineString, Point
import brightest_path_lib
from mapmanagercore.logger import logger

class AnnotationsSegments(SingleTimePointAnnotationsBase):
    def optimizeSegment(self, roughSegment: LineString, segment: LineString = None, updatedIdx: int = None, live: bool = False, 
                        z: int = None) -> Union[LineString, None]:
        if segment and len(roughSegment.coords) > 2:
            if updatedIdx > len(roughSegment.coords) - 1:
                if updatedIdx == 0:
                    return LineString([])
                return injectLine(segment, LineString([]), Point(roughSegment.coords[-1]), None)

            left = roughSegment.coords[updatedIdx -
                                       1] if updatedIdx > 0 else None
            point = roughSegment.coords[updatedIdx]
            right = roughSegment.coords[updatedIdx +
                                        1] if updatedIdx < len(roughSegment.coords) - 1 else None

            points = []
            if left:
                leftTracing = self.brightestPath(
                    LineString([left, point]), live, z)
                points = list(leftTracing.coords)
            if right:
                rightTracing = self.brightestPath(
                    LineString([point, right]), live, z)
                points.extend(rightTracing.coords)

            left = Point(left) if left else None
            right = Point(right) if right else None

            segment = injectLine(segment, LineString(
                points), left, right)
        else:
            segment = self.brightestPath(roughSegment, live, z)

        return segment.simplify(0.5)

    def brightestPath(self, roughSegment: LineString, live: bool = False, z: int = None):
        """
            Args:
                roughSegment:  LineString([left, point]) or  LineString([point, right])

        """
        # TODO: Add brightest path tracing
        # Limit tracing to the cube of the bounding box of the rough segment

        # if live:
        #     # TODO: Add brightest path tracing if it is fast enough to run in real time
        #     # TODO: Consider adding the mutation type along with the prior result if we can use it to speed things up
        #     return None

        zSpread = self.analysisParams.getValue('zSpread')
        channel = self.analysisParams.getValue('channel')

        #  channel: int, zRange: Tuple[int, int] = None, z: int = None, zSpread: int = 0
        image = self.getPixels(channel=channel, z=z, zSpread =zSpread).data() # returning in ndarray form
        if live:
            # brightest_path_lib.algorithm.AStarSearch()
            astar = brightest_path_lib.algorithm.AStarSearch(image, roughSegment[0], roughSegment[1])
            path = astar.search()
            logger.info(f"a start path: {path}")

        return roughSegment
