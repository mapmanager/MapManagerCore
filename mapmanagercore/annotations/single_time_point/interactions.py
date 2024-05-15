from typing import Union
import numpy as np
from shapely.geometry import Point
import shapely
from mapmanagercore.utils import shapeGrid
from .segment import AnnotationsSegments
from ...config import MAX_TRACING_DISTANCE, SegmentId, SpineId
from ...schemas import Segment, Spine
from ...layers.layer import DragState
from ...layers.utils import roundPoint
from shapely.geometry import LineString
from shapely.ops import split, linemerge
import geopandas as gp

from mapmanagercore.logger import logger

pendingBackgroundRoiTranslation = None


class AnnotationsInteractions(AnnotationsSegments):
    def nearestAnchor(self, segmentID: SegmentId,
                      point: Point,
                      brightestPathDistance: int = None,
                      channel: int = None,
                      zSpread: int = None):
        """
        Finds the nearest anchor point on a given line segment to a given point.

        Args:
            segmentID (SegmentId): The ID of the line segment.
            point (Point): The point to find the nearest anchor to.
            brightestPathDistance (int): The distance to search for the brightest path. Defaults to None.
            channel (int): The channel. Defaults to 0.
            zSpread (int): The z spread. Defaults to 0.

        Returns:
            Point: The nearest anchor point.
        """
        # abb
        # if not specified, get defaults from AnalysisParams()
        if brightestPathDistance is None:
            brightestPathDistance = self.analysisParams.getValue('brightestPathDistance')
        if channel is None:
            channel = self.analysisParams.getValue('channel')
        if zSpread is None:
            zSpread = self.analysisParams.getValue('zSpread')
            
        segment: LineString = self.segments[segmentID, "segment"]
        # find the closest point on the segment to the `point`
        minProjection = segment.project(point)

        if brightestPathDistance is None:
            # Default to the closest path
            anchor = segment.interpolate(minProjection)
            anchor = roundPoint(anchor, 1)
            return anchor

        segmentLength = int(segment.length)
        minProjection = int(minProjection)

        # create a range of distances to search for the brightest path
        range_ = range(
            max(minProjection-brightestPathDistance, 0),
            min(minProjection +
                brightestPathDistance + 1, segmentLength))

        # create a series of line segments from the point to anchors along the range
        targets = gp.GeoSeries([LineString(
            [point, roundPoint(segment.interpolate(distance), 1)]) for distance in range_])

        # get the pixel values for each line segment
        pixels = self.getShapePixels(
            targets, channel=channel, zSpread=zSpread)

        # Normalize the median brightness by the length of the path to pick the shortest brightest path
        brightest = (pixels.apply(np.median) / targets.length).idxmax()

        return Point(targets[brightest].coords[1])

    def snapBackgroundOffset(self, spineId: SpineId, channel: int = 0, zSpread: int = 0):
        roi = self.points[spineId, "roi"]
        z = self.points[spineId, "z"]

        # create a grid of points to search for the best offset
        grid = shapeGrid(roi, points=3, overlap=0.1)

        # translate the roi by the grid points
        candidates = gp.GeoSeries(grid.apply(
            lambda x: shapely.affinity.translate(roi, x["x"], x["y"]), axis=1))

        # get the pixel values for each candidate
        pixels = self.getShapePixels(
            candidates, channel=channel, zSpread=zSpread, z=z)

        # find the candidate with the lowest sum of pixel values
        offset = grid.iloc[pixels.apply(np.sum).idxmin()]

        # update the spine with the best offset
        self.updateSpine(spineId, Spine(
            xBackgroundOffset=offset["x"],
            yBackgroundOffset=offset["y"],
        ), replaceLog=True)

    def addSpine(self, segmentId: SpineId, x: int, y: int, z: int) -> Union[SpineId, None]:
        """
        Adds a spine.

        segmentId (str): The ID of the segment.
        x (int): The x coordinate of the spine.
        y (int): The y coordinate of the spine.
        z (int): The z coordinate of the spine.
        """
        point = Point(x, y, z)
        anchor = self.nearestAnchor(segmentId, point)
        spineId = self.newUnassignedSpineId()

        self.updateSpine(spineId, Spine.withDefaults(
            segmentID=segmentId,
            point=Point(point.x, point.y),
            z=int(z),
            anchor=Point(anchor.x, anchor.y),
            anchorZ=int(anchor.z),
            xBackgroundOffset=0.0,
            yBackgroundOffset=0.0,
        ))

        self.snapBackgroundOffset(spineId)

        return spineId

    def newUnassignedSpineId(self) -> SpineId:
        """
        Generates a new unique spine ID that is not assigned to any existing spine.

        Returns:
            int: new spine's ID.
        """
        ids = self._annotations.points.index.get_level_values(0)
        if len(ids) == 0:
            return 0
        return ids.max() + 1

    def moveSpine(self, spineId: SpineId, x: int, y: int, z: int, state: DragState = DragState.MANUAL) -> bool:
        """
        Moves the spine identified by `spineId` to the given `x` and `y` coordinates.

        Args:
            spineId (str): The ID of the spine to be translated.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.

        Returns:
            bool: True if the spine was successfully translated, False otherwise.
        """
        self.updateSpine(spineId, Spine(
            point=Point(x, y),
            z=z,
        ), state != DragState.START and state != DragState.MANUAL)

        return True

    def moveAnchor(self, spineId: SpineId, x: int, y: int, z: int, state: DragState = DragState.MANUAL) -> bool:
        """
        Moves the anchor point of a spine to the given x and y coordinates.

        Args:
            spineId (str): The ID of the spine.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.
            state (DragState): The state of the translation.

        Returns:
            bool: True if the anchor point was successfully translated, False otherwise.
        """
        segmentId = self.points[spineId, "segmentID"]
        anchor = self.nearestAnchor(segmentId, Point(x, y, z))

        logger.info(f'segmentId:{segmentId} anchor:{anchor}')

        self.updateSpine(spineId, Spine(
            anchorZ=int(anchor.z),
            anchor=Point(anchor.x, anchor.y),
        ), state != DragState.START and state != DragState.MANUAL)

        return True

    def moveBackgroundRoi(self, spineId: SpineId, x: int, y: int, z: int = 0, state: DragState = DragState.MANUAL) -> bool:
        """
        Translates the background ROI for a given spine ID by the specified x and y offsets.

        Args:
            spineId (str): The ID of the spine.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.
            state (DragState): The state of the translation.

        Returns:
            bool: True if the background ROI was successfully translated, False otherwise.
        """
        if state == DragState.MANUAL:
            self.updateSpine(spineId, Spine(
                xBackgroundOffset=float(x),
                yBackgroundOffset=float(y),
            ))
            return True

        point = self.points[spineId, [
            "xBackgroundOffset", "yBackgroundOffset"]]

        global pendingBackgroundRoiTranslation

        if pendingBackgroundRoiTranslation is None or state == DragState.START:
            pendingBackgroundRoiTranslation = [x, y]

        self.updateSpine(spineId, Spine(
            xBackgroundOffset=float(
                point["xBackgroundOffset"] + x - pendingBackgroundRoiTranslation[0]),
            yBackgroundOffset=float(
                point["yBackgroundOffset"] + y - pendingBackgroundRoiTranslation[1]),
        ), state != DragState.START and state != DragState.MANUAL)

        pendingBackgroundRoiTranslation = None if state == DragState.END else [
            x, y]

        return True

    def moveRoiExtend(self, spineId: SpineId, x: int, y: int, z: int = 0, state: DragState = DragState.MANUAL) -> bool:
        """
        Move the ROI extend for a given spine ID.

        Args:
            spineId (str): The ID of the spine.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.
            state (DragState): The state of the translation.

        returns:
            bool: True if the ROI extend was successfully translated, False otherwise.
        """

        point = self.points[spineId, "point"]

        self.updateSpine(spineId, Spine(
            roiExtend=float(point.distance(Point(x, y)))
        ), state != DragState.START and state != DragState.MANUAL)

        return True

    def moveRoiRadius(self, spineId: SpineId, x: int, y: int, z: int = 0, state: DragState = DragState.MANUAL) -> bool:
        """
        Move the ROI extend for a given spine ID.

        Args:
            spineId (str): The ID of the spine.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.
            state (DragState): The state of the translation.

        returns:
            bool: True if the ROI extend was successfully translated, False otherwise.
        """

        point = self.points[spineId, "point"]

        self.updateSpine(spineId, Spine(
            roiRadius=float(point.distance(Point(x, y)))
        ), state != DragState.START and state != DragState.MANUAL)

        return True

    def moveSegmentRadius(self, segmentId: SegmentId, x: int, y: int, z: int = 0, state: DragState = DragState.MANUAL) -> bool:
        """
        Move the Radius of a segment by the given x and y coordinates.

        Args:
            segmentId (str): The ID of the segment.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.
            state (DragState): The state of the translation.

        Returns:
            bool: True if the segment was successfully translated, False otherwise.
        """

        anchor = self.nearestAnchor(segmentId, Point(x, y))
        self.updateSegment(segmentId, Spine(
            radius=Point(anchor.x, anchor.y).distance(Point(x, y))
        ), state != DragState.START and state != DragState.MANUAL)

        return True

    # Segments

    def addSegment(self) -> Union[SegmentId, None]:
        """
        Generates a new segment.

        Args:
            t (int): The time point.

        Returns:
            int: The ID of the new segment.
        """
        segmentId = self.newUnassignedSegmentId()

        self.updateSegment(segmentId, Segment.withDefaults(
            segment=LineString([]),
            roughTracing=LineString([])
        ))

        return segmentId

    def newUnassignedSegmentId(self) -> SegmentId:
        """
        Generates a new unique segment ID that is not assigned to any existing segment.

        Returns:
            int: new segment's ID.
        """
        ids = self._annotations.segments.index.get_level_values(0)
        if len(ids) == 0:
            return 0
        return ids.max() + 1

    def appendSegmentPoint(self, segmentId: SegmentId, x: int, y: int, z: int, speculate: bool = False) -> LineString:
        """
        Adds a point to a segment.

        Args:
            segmentId (str): The ID of the segment.
            x (int): The x coordinate of the point.
            y (int): The y coordinate of the point.
            z (int): The z coordinate of the point.
            speculate (bool): Whether to simulate the addition without actually adding the point. Defaults to False.

        Returns:
            LineString: The updated rough tracing.
        """

        roughTracing: LineString = self.segments[segmentId,
                                                 "roughTracing"]
        point = Point(x, y, z)
        snappedPoint = roughTracing.interpolate(roughTracing.project(point))

        if MAX_TRACING_DISTANCE is not None and point.distance(snappedPoint) > MAX_TRACING_DISTANCE:
            # Snap the point to the maximum tracing distance
            point = LineString([snappedPoint.coords[0], point.coords[0]]).interpolate(
                MAX_TRACING_DISTANCE)

        if roughTracing.coords[0] == snappedPoint.coords[0]:
            # Prepend the point to the rough tracing
            roughTracing = LineString(
                [point.coords[0]] + list(roughTracing.coords))
        elif roughTracing.coords[-1] == snappedPoint.coords[0]:
            # Append the point to the rough tracing
            roughTracing = LineString(
                list(roughTracing.coords) + [point.coords[0]])
        else:
            # Add a new point on the rough tracing
            roughTracing = linemerge(split(roughTracing, snappedPoint).geoms)

        if speculate:
            return roughTracing

        self.updateSegmentWithLiveTracing(segmentId, roughTracing)
        return roughTracing

    def moveSegmentPoint(self, segmentId: SegmentId, x: int, y: int, z: int, index: int, state: DragState = DragState.MANUAL) -> bool:
        """
        Moves a point in a segment.

        Args:
            segmentId (str): The ID of the segment.
            x (int): The x coordinate of the point.
            y (int): The y coordinate of the point.
            z (int): The z coordinate of the point.
            index (int): The index of the point to move.
        """
        roughTracing = list(
            self.segments[segmentId, "roughTracing"].coords)

        roughTracing[index] = (x, y, z)
        self.updateSegmentWithLiveTracing(
            segmentId, LineString(roughTracing), state != DragState.START and state != DragState.MANUAL)

        return True

    def deleteSegmentPoint(self, segmentId: SegmentId, index: int) -> bool:
        """
        Deletes a point from a segment.

        Args:
            segmentId (str): The ID of the segment.
            index (int): The index of the point to delete.
        """
        roughTracing = list(
            self.segments[segmentId, "roughTracing"].coords)

        del roughTracing[index]
        self.updateSegmentWithLiveTracing(segmentId, LineString(roughTracing))

        return True

    def updateSegmentWithLiveTracing(self, segmentId: SegmentId, roughTracing: LineString, replaceLog: bool = False):
        """
        Updates a segment with live tracing.

        Args:
            segmentId (str): The ID of the segment.
            roughTracing (LineString): The rough tracing.
        """
        segment = self.optimizeSegment(roughTracing, live=True)

        update = Segment(roughTracing=roughTracing)

        if segment is not None:
            update.segment = segment

        self.updateSegment(segmentId, update, replaceLog)

    def commitSegmentTracing(self, segmentId: SegmentId) -> bool:
        """
        Commits the rough tracing of a segment.

        Args:
            segmentId (str): The ID of the segment.
        """

        roughTracing = self.segments[segmentId, "roughTracing"]
        self.updateSegment(segmentId, Segment(
            segment=self.optimizeSegment(roughTracing)
        ), skipLog=True)

        return True
