from typing import Union
from .query import QueryAnnotations
from shapely.geometry import LineString


class AnnotationsSegments(QueryAnnotations):
    def optimizeSegment(self, roughSegment: LineString, live: bool = False) -> Union[LineString, None]:
        if live:
            # TODO: Add brightest path tracing if it is fast enough to run in real time
            # TODO: Consider adding the mutation type along with the prior result if we can use it to speed things up
            return None

        # TODO: Add brightest path tracing

        return roughSegment
