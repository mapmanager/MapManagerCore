from typing import Union
from mapmanagercore.utils import injectLine
from .base import SingleTimePointAnnotationsBase
from shapely.geometry import LineString, Point


class AnnotationsSegments(SingleTimePointAnnotationsBase):
    def optimizeSegment(self, roughSegment: LineString, segment: LineString = None, updatedIdx: int = None, live: bool = False) -> Union[LineString, None]:
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
                    LineString([left, point]), live)
                points = list(leftTracing.coords)
            if right:
                rightTracing = self.brightestPath(
                    LineString([point, right]), live)
                points.extend(rightTracing.coords)

            left = Point(left) if left else None
            right = Point(right) if right else None

            segment = injectLine(segment, LineString(
                points), left, right)
        else:
            segment = self.brightestPath(roughSegment, live)

        return segment.simplify(0.5)

    def brightestPath(self, roughSegment: LineString, live: bool = False):
        # TODO: Add brightest path tracing
        # Limit tracing to the cube of the bounding box of the rough segment

        # if live:
        #     # TODO: Add brightest path tracing if it is fast enough to run in real time
        #     # TODO: Consider adding the mutation type along with the prior result if we can use it to speed things up
        #     return None

        return roughSegment
