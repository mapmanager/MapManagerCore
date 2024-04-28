from typing import Tuple, Union
import numpy as np
from shapely.geometry import Point
from .segment import AnnotationsSegments
from ..config import MAX_TRACING_DISTANCE, SegmentId, Spine, SpineId, Segment
from ..layers.layer import DragState
from ..layers.utils import roundPoint
from shapely.geometry import LineString
from shapely.ops import split, linemerge

from mapmanagercore.logger import logger


class AnnotationsInteractions(AnnotationsSegments):
    def nearestAnchor(self,
                      segmentID: Tuple[SegmentId, int],
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
        
        # logger.info('')

        # if not specified, get defaults from AnalysisParams()
        if brightestPathDistance is None:
            brightestPathDistance = self._analysisParams.getValue('brightestPathDistance')
        if channel is None:
            channel = self._analysisParams.getValue('channel')
        if zSpread is None:
            zSpread = self._analysisParams.getValue('zSpread')

        # logger.warning(f'brightestPathDistance:{brightestPathDistance} channel:{channel} zSpread:{zSpread}')

        segment: LineString = self._lineSegments.loc[segmentID, "segment"]
        minProjection = segment.project(point)

        if brightestPathDistance is not None:
            segmentLength = int(segment.length)
            minProjection = int(minProjection)
            range_ = range(
                max(minProjection-brightestPathDistance, 0),
                min(minProjection +
                    brightestPathDistance + 1, segmentLength))

            targets = [LineString([point, roundPoint(segment.interpolate(
                minProjection + distance), 1)]) for distance in range_]

            brightest = self.getShapePixels(
                targets, channel=channel, zSpread=zSpread, time=segmentID[1]).apply(np.sum).idxmax()
            return Point(targets[brightest].coords[1])

        anchor = segment.interpolate(minProjection)
        anchor = roundPoint(anchor, 1)
        return anchor

    def connect(self, spineKey: Tuple[SpineId, int], toSpineKey: Tuple[SpineId, int]):
        if self._points.loc[toSpineKey, "segmentID"] != self._points.loc[spineKey, "segmentID"]:
            raise ValueError("Cannot connect spines from different segments.")

        # check if the key already exists in the time point
        existingKey = (toSpineKey[0], spineKey[0])
        if existingKey in self._points.index:
            self.disconnect(existingKey)

        # Propagate the spine ID to all future time points
        self.updateSpine(range(spineKey, spineKey[0]), {
            "spineID": toSpineKey[0],
        })

    def disconnect(self, spineKey: Tuple[SpineId, int]):
        newID = self.newUnassignedSpineId()

        # Propagate the spine ID change to all future time points
        self.updateSpine(range(spineKey, spineKey[0]), {
            "spineID": newID,
        })

    def addSpine(self,
                 segmentId: Tuple[SpineId, int],
                 x: int,
                 y: int,
                 z: int,
                 brightestPathDistance: int = None,
                 channel: int = None,
                 zSpread: int = None
                 ) -> Union[SpineId, None]:
        """
        Adds a spine.

        segmentId (str): The ID of the segment.
        x (int): The x coordinate of the spine.
        y (int): The y coordinate of the spine.
        z (int): The z coordinate of the spine.
        """
        point = Point(x, y, z)
        anchor = self.nearestAnchor(segmentId,
                                    point,
                                    brightestPathDistance,
                                    channel=channel,
                                    zSpread=zSpread)
        spineId = self.newUnassignedSpineId()

        self.updateSpine((spineId, segmentId[1]), {
            **Spine.defaults(),
            "segmentID": segmentId[0],
            "point": Point(point.x, point.y),
            "z": int(z),
            "anchor": Point(anchor.x, anchor.y),
            "anchorZ": int(anchor.z),
            "xBackgroundOffset": 0.0,
            "yBackgroundOffset": 0.0,
            "roiExtend": self._analysisParams['roiExtend'],
            "roiRadius": self._analysisParams['roiRadius'],
        })

        return spineId

    def newUnassignedSpineId(self) -> SpineId:
        """
        Generates a new unique spine ID that is not assigned to any existing spine.

        Returns:
            int: new spine's ID.
        """
        ids = self._points.index.get_level_values(0)
        if len(ids) == 0:
            return 0
        return self._points.index.get_level_values(0).max() + 1

    def moveSpine(self, spineId: Tuple[SpineId, int], x: int, y: int, z: int, state: DragState = DragState.MANUAL) -> bool:
        """
        Moves the spine identified by `spineId` to the given `x` and `y` coordinates.

        Args:
            spineId (str): The ID of the spine to be translated.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.

        Returns:
            bool: True if the spine was successfully translated, False otherwise.
        """
        self.updateSpine(spineId, {
            "point": Point(x, y),
            "z": z,
        }, state != DragState.START and state != DragState.MANUAL)

        return True

    def moveAnchor(self, spineId: Tuple[SpineId, int], x: int, y: int, z: int, state: DragState = DragState.MANUAL) -> bool:
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
        point = self._points.loc[spineId]
        anchor = self.nearestAnchor(point["segmentID"], Point(x, y))

        self.updateSpine(spineId, {
            "anchorZ": int(anchor.z),
            "anchor": Point(anchor.x, anchor.y),
        }, state != DragState.START and state != DragState.MANUAL)

        return True

    pendingBackgroundRoiTranslation = None

    def moveBackgroundRoi(self, spineId: Tuple[SpineId, int], x: int, y: int, z: int = 0, state: DragState = DragState.MANUAL) -> bool:
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
            self.updateSpine(spineId, {
                "xBackgroundOffset": float(x),
                "yBackgroundOffset": float(y),
            })
            return True

        point = self._points.loc[spineId]

        if self.pendingBackgroundRoiTranslation is None or state == DragState.START:
            self.pendingBackgroundRoiTranslation = [x, y]

        self.updateSpine(spineId, {
            "xBackgroundOffset": float(point["xBackgroundOffset"] + x - self.pendingBackgroundRoiTranslation[0]),
            "yBackgroundOffset": float(point["yBackgroundOffset"] + y - self.pendingBackgroundRoiTranslation[1]),
        }, state != DragState.START and state != DragState.MANUAL)

        self.pendingBackgroundRoiTranslation = [x, y]

        if state == DragState.END:
            self.pendingBackgroundRoiTranslation = None

        return True

    def moveRoiExtend(self, spineId: Tuple[SpineId, int], x: int, y: int, z: int = 0, state: DragState = DragState.MANUAL) -> bool:
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

        point = self._points.loc[spineId, "point"]

        self.updateSpine(spineId, {
            "roiExtend": float(point.distance(Point(x, y)))
        }, state != DragState.START and state != DragState.MANUAL)

        return True

    def moveSegmentRadius(self, segmentId: Tuple[SegmentId, int], x: int, y: int, z: int = 0, state: DragState = DragState.MANUAL) -> bool:
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

        anchor = self.nearestAnchor(segmentId, Point(x, y), True)
        self.updateSegment(segmentId, {
            "radius": Point(anchor.x, anchor.y).distance(Point(x, y))
        }, state != DragState.START and state != DragState.MANUAL)

        return True

    # Segments

    def connectSegment(self, segmentKey: Tuple[SegmentId, int], toSegmentKey: Tuple[SegmentId, int]):
        if segmentKey[1] == toSegmentKey[1]:
            raise ValueError(
                "Cannot connect segments in the same time points.")

        # check if the key already exists in the time point
        existingKey = (toSegmentKey[0], segmentKey[1])
        if existingKey in self._lineSegments.index:
            self.disconnectSegment(existingKey)

        # Propagate the segment ID to all future time points
        self.updateSegment(range(segmentKey, segmentKey[0]), {
            "segmentID": toSegmentKey[0],
        })

    def disconnectSegment(self, segmentKey: Tuple[SegmentId, int]):
        newID = self.newUnassignedSegmentId()

        # Propagate the segment ID change to all future time points
        self.updateSegment(range(segmentKey, segmentKey[0]), {
            "segmentID": newID,
        })

    def addSegment(self, t: int = 0) -> Union[SegmentId, None]:
        """
        Generates a new segment.

        Args:
            t (int): The time point.

        Returns:
            int: The ID of the new segment.
        """
        segmentId = self.newUnassignedSegmentId()

        self.updateSegment((segmentId, t), {
            **Segment.defaults(),
            # "segmentID": segmentId,
            "segment": LineString([]),
            "roughTracing": LineString([])
        })

        return segmentId

    def newUnassignedSegmentId(self) -> SegmentId:
        """
        Generates a new unique segment ID that is not assigned to any existing segment.

        Returns:
            int: new segment's ID.
        """
        ids = self._lineSegments.index.get_level_values(0)
        if len(ids) == 0:
            return 0
        return self._lineSegments.index.get_level_values(0).max() + 1

    def appendSegmentPoint(self, segmentId: Tuple[SegmentId, int], x: int, y: int, z: int, speculate: bool = False) -> LineString:
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

        roughTracing: LineString = self._lineSegments.loc[segmentId,
                                                          "roughTracing"]

        # abb
        # _numPoints = len(self._lineSegments)
        # logger.info(f'_numPoints:{_numPoints}')

        point = Point(x, y, z)
        snappedPoint = roughTracing.interpolate(roughTracing.project(point))
        
        if MAX_TRACING_DISTANCE is not None and point.distance(snappedPoint) > MAX_TRACING_DISTANCE:
            # Snap the point to the maximum tracing distance
            point = LineString([snappedPoint.coords[0], point.coords[0]]).interpolate(
                MAX_TRACING_DISTANCE)

        # if _numPoints:
        #     logger.info('xxx')
        #     _twoPoints = [(x, y, z),(x, y, z)]
        #     roughTracing = LineString(_twoPoints)  # list[tuple]
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

    def moveSegmentPoint(self, segmentId: Tuple[SegmentId, int], x: int, y: int, z: int, index: int, state: DragState = DragState.MANUAL) -> bool:
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
            self._lineSegments.loc[segmentId, "roughTracing"].coords)

        roughTracing[index] = (x, y, z)
        self.updateSegmentWithLiveTracing(
            segmentId, LineString(roughTracing), state != DragState.START and state != DragState.MANUAL)

        return True

    def deleteSegmentPoint(self, segmentId: Tuple[SegmentId, int], index: int) -> bool:
        """
        Deletes a point from a segment.

        Args:
            segmentId (str): The ID of the segment.
            index (int): The index of the point to delete.
        """
        roughTracing = list(
            self._lineSegments.loc[segmentId, "roughTracing"].coords)

        del roughTracing[index]
        self.updateSegmentWithLiveTracing(segmentId, LineString(roughTracing))

        return True

    def updateSegmentWithLiveTracing(self, segmentId: Tuple[SegmentId, int], roughTracing: LineString, replaceLog: bool = False):
        """
        Updates a segment with live tracing.

        Args:
            segmentId (str): The ID of the segment.
            roughTracing (LineString): The rough tracing.
        """
        segment = self.optimizeSegment(roughTracing, live=True)

        update = {
            "roughTracing": roughTracing
        }

        if segment is not None:
            update["segment"] = segment

        self.updateSegment(segmentId, update, replaceLog)

    def commitSegmentTracing(self, segmentId: Tuple[SegmentId, int]) -> bool:
        """
        Commits the rough tracing of a segment.

        Args:
            segmentId (str): The ID of the segment.
        """

        roughTracing = self._lineSegments.loc[segmentId, "roughTracing"]
        self.updateSegment(segmentId, {
            "segment": self.optimizeSegment(roughTracing)
        }, skipLog=True)

        return True
