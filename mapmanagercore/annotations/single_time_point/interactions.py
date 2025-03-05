from typing import Union
import numpy as np
from shapely.geometry import Point
import shapely
from mapmanagercore.utils import injectPoint, shapeGrid
from .segment import AnnotationsSegments
from ...config import SegmentId, SpineId
from ...schemas import Segment, Spine
from ...layers.layer import DragState
from ...layers.utils import roundPoint
from shapely.geometry import LineString
import geopandas as gp
from mapmanagercore.logger import logger
from mapmanagercore.logger import logger

pendingBackgroundRoiTranslation = None


class AnnotationsInteractions(AnnotationsSegments):

    def getSpineDistance(self, segmentID: SegmentId,
                         point: Point):
        """Add doc string
        """
        segment: LineString = self.segments[segmentID, "segment"]
        # find the closest point on the segment to the `point`
        minProjection = segment.project(point)
        return minProjection

    # abb added findBrightest=False, do not find brightest by default
    # abj added returnIndex=False, return Anchor point by default, return its index if True
    def nearestAnchor(self, segmentID: SegmentId,
                      point: Point,
                      findBrightest: bool = False,
                      returnIndex: bool = False
                      ):
        """Finds the nearest anchor point on a given line segment to a given point.

        Args:
            segmentID (SegmentId): The ID of the line segment.
            point (Point): The point to find the nearest anchor to.
            findBrightest (bool): Default False.
                If True then find the brightest anchor using image data.

        Returns:
            Point: The nearest anchor point (not required to be in original)
        """
        segment: LineString = self.segments[segmentID, "segment"]

        # find the closest point on the segment to the `point`
        # logger.warning('=== PROJECTING')
        # logger.warning(f'  segmentID:{segmentID}')
        # logger.warning(f'  segment:{segment}')
        # logger.warning(f'  point:{point}')

        minProjection = segment.project(point)

        # abb debug
        if np.isnan(minProjection):
            logger.error(f'=== UNEXPECTED minProjection:{minProjection}')
            logger.error(f'   segmentID:{segmentID}')
            # logger.error(f'   self.segments[:]:{self.segments[:]}')
            logger.error(f'   segment:{segment}')
            logger.error(f'   point:{point}')

        if not findBrightest:
            # Default to the closest point (not brightest)
            # problem: closest point on line but not actually a point stored within segments
            logger.info("defaulting to closest point")
            anchor = segment.interpolate(minProjection)
            anchor = roundPoint(anchor, 1)
            logger.info(f"  anchor {anchor}")

            if returnIndex:
                logger.info(f"entering return index")
                distanceStart = None
                # find index of point in segment that is closest to minProjection
                x, y = segment.xy
                for i, val in enumerate(x):
                    currentPoint = Point(x[i], y[i])
                    logger.info(f"currentPoint {currentPoint}")
                    # check the distance
                    distanceCheck = Point(
                        anchor.x, anchor.y).distance(currentPoint)
                    if distanceStart is None:
                        distanceStart = distanceCheck

                    # check for closer point, where distance is less
                    if abs(distanceCheck) < abs(distanceStart):
                        logger.info(
                            f"distanceCheck {abs(distanceCheck)} < distanceStart {abs(distanceStart)}")
                        pivotIndex = i
                        distanceStart = distanceCheck
                        logger.info(f"pivotIndex {pivotIndex}")

                return pivotIndex

            # otherwise return anchor
            return anchor

        brightestPathDistance = self.analysisParams[
            'brightestPathDistance']
        channel = self.analysisParams['channel']
        zSpread = self.analysisParams['zSpread']

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

    # abj
    def autoConnectBrightestIndex(self, spineId: SpineId,
                                  segmentID: SegmentId,
                                  point: Point,
                                  findBrightest: bool = True,
                                  ):
        """ Calculates nearest (brightest) anchor and sets it

        Args:
            segmentID (SegmentId): The ID of the line segment.
            point (Point): The point to find the nearest anchor to.


        """
        anchor = self.nearestAnchor(segmentID, point, findBrightest)

        self.updateSpine(spineId, Spine(
            anchorZ=int(anchor.z),
            anchor=Point(anchor.x, anchor.y),
        ), replaceLog=True)

        return True

    def snapBackgroundOffset(self, spineId: SpineId,
                             channel: int = None,
                             zSpread: int = None):

        # abb analysisparams
        if channel is None:
            channel = self.analysisParams['channel']
        if zSpread is None:
            zSpread = self.analysisParams['zSpread']

        # abb 20241221 after s-dev merge -->> ERROR
        roi = self.points[spineId, "roi"]
        
        # logger.error(f'spineId:{spineId} roi is:')
        # print(type(roi))
        # print(roi)

        z = self.points[spineId, "z"]

        # create a grid of points to search for the best offset
        points = self.analysisParams['backgroundRoiGridPoints']
        overlap = self.analysisParams['backgroundRoiGridOverlap']

        try:
            grid = shapeGrid(roi, points=points, overlap=overlap)  # abj
            # grid = shapeGrid(roi, points=3, overlap=0.1)
        except (ValueError) as e:
            logger.error(f'   {e}')
            logger.error(f'   spineId:{spineId}')
            logger.error(f'   roi:{roi}')
            print('   self.points[:]')
            print(self.points[:])
            return

        # translate the roi by the grid points
        candidates = gp.GeoSeries(grid.apply(
            lambda x: shapely.affinity.translate(roi, x["x"], x["y"]), axis=1))

        # get the pixel values for each candidate
        pixels = self.getShapePixels(
            candidates, channel=channel, zSpread=zSpread, z=z)

        # TODO: exclude out of bounds candidates

        # find the candidate with the lowest sum of pixel values
        offset = grid.iloc[pixels.apply(np.sum).idxmin()]

        # update the spine with the best offset
        self.updateSpine(spineId, Spine(
            xBackgroundOffset=offset["x"],
            yBackgroundOffset=offset["y"],
        ), replaceLog=True)

    def setSegmentOrigin(self, segmentId: SegmentId, x: int, y: int, z: int, replaceLog=False) -> bool:
        """
        Sets the origin of the segment

        segmentId (str): The ID of the segment.
        x (int): The x coordinate of a point near the new origin.
        y (int): The y coordinate of a point near the new origin.
        z (int): The z coordinate of a point near the new origin.
        """
        point = Point(x, y, z)

        if not segmentId in self.segments.index:
            logger.warning(f'segmentId:{segmentId} not in self.segments.index')
            return False

        segment: LineString = self.segments[segmentId, "segment"]

        pivotDistance = segment.project(point)
        self.updateSegment(segmentId, Segment(
            pivotDistance=pivotDistance
        ), replaceLog=replaceLog)

        return True

    def addSpine(self, segmentId: SegmentId, x: int, y: int, z: int) -> Union[SpineId, None]:
        """
        Adds a spine.

        segmentId (str): The ID of the segment.
        x (int): The x coordinate of the spine.
        y (int): The y coordinate of the spine.
        z (int): The z coordinate of the spine.
        """
        if not self._inBounds(x, y, z):
            logger.warning(f'x:{x} y:{y} z:{z} is out of bound -> no new spine')
            return None

        point = Point(x, y, z)
         
        if segmentId not in self.segments.index:
            logger.warning(f'segmentId:{segmentId} not in self.segments.index')
            return None

        # logger.error(f'1 FutureWarning: The `drop` keyword ...')
        anchor = self.nearestAnchor(segmentId, point, findBrightest=True)

        spineId = self.newUnassignedSpineId()

        # abb TODO we want our spine id(s) to be Python int, not numpy int64 ???
        spineId = int(spineId)

        _spine: Spine = Spine(
            segmentID=segmentId,
            point=Point(point.x, point.y),
            z=int(z),
            anchor=Point(anchor.x, anchor.y),
            anchorZ=int(anchor.z),
            xBackgroundOffset=0.0,
            yBackgroundOffset=0.0,
            roiExtend=self.analysisParams["roiExtend"],
            roiRadius=self.analysisParams["roiRadius"]
        ).defaults()
        self.updateSpine(spineId, _spine)

        self.snapBackgroundOffset(spineId)

        return spineId

    def newUnassignedSpineId(self) -> SpineId:
        """
        Generates a new unique spine ID that is not assigned to any existing spine.

        Returns:
            int: new spine's ID.
        """
        return self._annotations.newUnassignedSpineId()

    def _inBounds(self, x: int, y: int, z: int) -> bool:
        """
        Checks if the given coordinates are within the bounds of the image.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.
            z (int): The z-coordinate.

        Returns:
            bool: True if the coordinates are within the bounds of the image, False otherwise.
        """
        sz, sx, sy = self.shape
        return x >= 0 and y >= 0 and z >= 0 and x < sx and y < sy and z < sz

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

        if not self._inBounds(x, y, z):
            return False

        oldPoint = self.points[spineId, "point"]
        oldZ = self.points[spineId, "z"]
        wasInBounds = self.points[spineId, "isValid"]

        self.updateSpine(spineId, Spine(
            point=Point(x, y),
            z=z,
        ), state != DragState.START and state != DragState.MANUAL)

        if wasInBounds and not self.points[spineId, "isValid"]:
            self.updateSpine(spineId, Spine(
                point=oldPoint,
                z=oldZ
            ), True)
            return False

        return True

    def moveAnchor(self, spineId: SpineId,
                   x: int, y: int, z: int,
                   state: DragState = DragState.MANUAL) -> bool:
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

        # abb when moving, do not find brightest
        anchor = self.nearestAnchor(segmentId, Point(x, y, z))

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

        wasInBounds = self.points[spineId, "roiBgInBounds"]
        point = self.points[spineId, [
            "xBackgroundOffset", "yBackgroundOffset"]]

        global pendingBackgroundRoiTranslation

        if pendingBackgroundRoiTranslation is None or state == DragState.START:
            pendingBackgroundRoiTranslation = [x, y]

        oldXBackgroundOffset = point["xBackgroundOffset"]
        oldYBackgroundOffset = point["yBackgroundOffset"]
        self.updateSpine(spineId, Spine(
            xBackgroundOffset=float(
                oldXBackgroundOffset + x - pendingBackgroundRoiTranslation[0]),
            yBackgroundOffset=float(
                oldYBackgroundOffset + y - pendingBackgroundRoiTranslation[1]),
        ), state != DragState.START and state != DragState.MANUAL)

        # Revert to the previous offset if the Bg-ROI is out of bounds
        if wasInBounds and not self.points[spineId, "roiBgInBounds"]:
            self.updateSpine(spineId, Spine(
                xBackgroundOffset=oldXBackgroundOffset,
                yBackgroundOffset=oldYBackgroundOffset,
            ), True)

        pendingBackgroundRoiTranslation = None if state == DragState.END else [
            x, y]

        return True

    def moveRoiExtend(self, spineId: SpineId, x: int, y: int, z: int = 0,
                      state: DragState = DragState.MANUAL, roiExtend: int = None) -> bool:
        """
        Move the ROI extend for a given spine ID.

        Args:
            spineId (str): The ID of the spine.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.
            state (DragState): The state of the translation.
            roiExtend (int): updates roiExtend for Spine if given a value (#abj)
                - used to standardize the roiExtend of a point across multiple time points

        returns:
            bool: True if the ROI extend was successfully translated, False otherwise.
        """

        point = self.points[spineId, "point"]

        if roiExtend is None:
            roiExtend = float(point.distance(Point(x, y)))

        oldRoiExtend = self.points[spineId, "roiExtend"]
        wasInBounds = self.points[spineId, "isValid"]
        self.updateSpine(spineId, Spine(
            roiExtend=float(roiExtend)
        ), state != DragState.START and state != DragState.MANUAL)

        # Revert to the previous offset if the ROI is out of bounds
        if wasInBounds and not self.points[spineId, "isValid"]:
            self.updateSpine(spineId, Spine(
                roiExtend=oldRoiExtend
            ), True)

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

        wasInBounds = self.points[spineId, "isValid"]
        oldRoiRadius = self.points[spineId, "roiRadius"]

        self.updateSpine(spineId, Spine(
            roiRadius=float(point.distance(Point(x, y)))
        ), state != DragState.START and state != DragState.MANUAL)

        # Revert to the previous offset if the ROI is out of bounds
        if wasInBounds and not self.points[spineId, "isValid"]:
            self.updateSpine(spineId, Spine(
                roiRadius=oldRoiRadius
            ), True)

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
        self.updateSegment(segmentId, Segment(
            radius=Point(anchor.x, anchor.y).distance(Point(x, y))
        ), state != DragState.START and state != DragState.MANUAL)

        return True

    # Segments

    def newSegment(self) -> Union[SegmentId, None]:
        """
        Generates a new segment.

        Args:
            t (int): The time point.

        Returns:
            int: The ID of the new segment.
        """
        segmentId = self.newUnassignedSegmentId()
        segmentId = int(segmentId)

        _segment: Segment = Segment(
            segment=LineString([]),
            roughTracing=LineString([]),
            radius=self.analysisParams["segmentRadius"]
        ).defaults()

        self.updateSegment(segmentId, _segment)

        return segmentId

    def newUnassignedSegmentId(self) -> SegmentId:
        """
        Generates a new unique segment ID that is not assigned to any existing segment.

        Returns:
            int: new segment's ID.
        """
        return self._annotations.newUnassignedSegmentId()

    def injectSegmentPoint(self, segmentId: SegmentId, x: int, y: int, z: int):
        segment: LineString = self.segments[segmentId, "segment"]
        roughTracing: LineString = self.segments[segmentId, "roughTracing"]
        snappedPoint = segment.interpolate(segment.project(Point(x, y, z)))
        snappedPoint = Point(snappedPoint.x, snappedPoint.y, z)
        roughTracing, idx = injectPoint(roughTracing, snappedPoint)

        if roughTracing is None:
            return None

        self.updateSegmentWithLiveTracing(segmentId, roughTracing.coords, idx)
        return idx

    def appendSegmentPoint(self, segmentId: SegmentId,
                           x: int, y: int, z: int,
                           speculate: bool = False) -> LineString:
        """Adds a point to a segment.

        Args:
            segmentId (str): The ID of the segment.
            x (int): The x coordinate of the point.
            y (int): The y coordinate of the point.
            z (int): The z coordinate of the point.
            speculate (bool): Whether to simulate the addition without actually adding the point. Defaults to False.

        Returns:
            LineString: The updated rough tracing.
        """
        
        if segmentId not in self.segments.index:
            # Create a new segment if it doesn't exist
            self.updateSegment(segmentId, Segment(
                segment=LineString([]),
                roughTracing=LineString([])
            ).defaults())

        # abb debug
        # logger.info(f'segmentId:{segmentId} {type(segmentId)}')
        # print('   self.segments:')
        # # self.segments is mapmanagercore.annotations.single_time_point.base.SingleTimePointFrame
        # print(self.segments)

        roughTracing: Union[LineString,
                            Point] = self.segments[segmentId, "roughTracing"]

        # abb debug
        # roughTracing is LINESTRING Z
        if roughTracing is None:
            logger.error(
                f'   segmentId:{segmentId} roughTracing IS NONE -->> ERROR')
            logger.error('self.segments.index:')
            logger.error(self.segments.index)

        point = Point(x, y, z)
        
        # abb what is first? It seems to always be true?
        first = len(roughTracing.coords) < 2 or point.distance(
            Point(roughTracing.coords[0])) < point.distance(Point(roughTracing.coords[-1]))
        
        snappedPoint = point if len(roughTracing.coords) == 0 else Point(
            roughTracing.coords[0 if first else -1])

        maxTracingDistance = self.analysisParams.getValue(
            "segmentTracingMaxDistance")

        # logger.warning(f'abb')
        # logger.info(F'   roughTracing;{roughTracing}')
        # logger.info(F'   roughTracing;{roughTracing.coords}')
        # logger.info(f'   point:{point}')
        # logger.info(f'   first:{first}')
        # if len(roughTracing.coords) > 0:
        #     logger.info(f'      {point.distance(Point(roughTracing.coords[0]))}')
        #     logger.info(f'      {point.distance(Point(roughTracing.coords[-1]))}')
        # logger.info(f'   snappedPoint:{snappedPoint}')
        # logger.info(f'   maxTracingDistance:{maxTracingDistance}')

        # logger.warning(f'abb')
        # logger.info(F'   roughTracing;{roughTracing}')
        # logger.info(F'   roughTracing;{roughTracing.coords}')
        # logger.info(f'   point:{point}')
        # logger.info(f'   first:{first}')
        # if len(roughTracing.coords) > 0:
        #     logger.info(f'      {point.distance(Point(roughTracing.coords[0]))}')
        #     logger.info(f'      {point.distance(Point(roughTracing.coords[-1]))}')
        # logger.info(f'   snappedPoint:{snappedPoint}')
        # logger.info(f'   maxTracingDistance:{maxTracingDistance}')

        if maxTracingDistance is not None and point.distance(snappedPoint) > maxTracingDistance:
            logger.warning(f'abb return None for maxTracingDistance:{maxTracingDistance}')
            return None

        if first:
            # abb always true?
            if speculate:
                return self.optimizeSegment(LineString([
                    point.coords[0],
                    roughTracing.coords[0] if len(
                        roughTracing.coords) > 0 else point.coords[0]
                ]), live=True)

            # Prepend the point to the rough tracing
            roughTracing = [point.coords[0], *roughTracing.coords]
            idx = 0
        else:
            # Append the point to the rough tracing
            if speculate:
                return self.optimizeSegment(LineString([
                    roughTracing.coords[-1] if len(
                        roughTracing.coords) > 0 else point.coords[0],
                    point.coords[0],
                ]), live=True)

            logger.warning(f'abb point;{point}')
            logger.warning(f'  point.coords:{point.coords}')
            
            # Append the point to the rough tracing
            roughTracing = [*roughTracing.coords, point.coords[0]]
            idx = len(roughTracing) - 1

        # logger.info(f'idx:{idx} roughTracing:{roughTracing}')

        self.updateSegmentWithLiveTracing(
            segmentId, roughTracing, idx)

        theRet = 0 if first else len(roughTracing) - 1
        # logger.info(f'theRet:{theRet}')
        return theRet

    # Use setSegmentOrigin instead
    # def old_setPivotPoint(self, segmentId: SegmentId, clickedPoint: Point, speculate: bool = False) -> Point:
    #     """ Sets pivotPoint of segment.

    #     Calculates pivot point by find closest brightest index point to the clicked Point
    #     """
    #     # pass

    #     # auto set pivot point closest to click on line

    #     closestPivotPoint = self.nearestAnchor(segmentID=segmentId, point=clickedPoint, findBrightest=False)
    #     logger.info(f"closestPivotPoint {closestPivotPoint}")
    #     # set pivot point in backend
    #     self.updateSegment(segmentId, Segment(
    #         pivotPoint=closestPivotPoint
    #     ))

    #     # used for verification
    #     return closestPivotPoint

    # Use setSegmentOrigin instead
    # def setPivotDistance(self, segmentId: SegmentId, clickedPoint: Point, speculate: bool = False) -> Point:
    #     """ Sets pivotPoint of segment.

        Given a point, finds the distance along the line to that (projected) point.
        """
        
        segment: LineString = self.segments[segmentId, "segment"]

        # line_locate_point: Returns the distance to the line origin of given point.
        #  if point is not on line, first applies project
        pivotDistance = shapely.line_locate_point(segment, clickedPoint)

        logger.info(f'segmentId:{segmentId} pivotDistance:{pivotDistance}')

        # reverse back to point
        _reversePoint = shapely.line_interpolate_point(segment, pivotDistance)
        logger.info(f'   _reversePoint:{_reversePoint}')

        # set pivot point in backend
        self.updateSegment(segmentId, Segment(
            pivotDistance = pivotDistance
        ))

        return pivotDistance

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
            segmentId, roughTracing, index, replaceLog=state != DragState.START and state != DragState.MANUAL)

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

        self.updateSegmentWithLiveTracing(segmentId, roughTracing, index)

        if len(roughTracing) == 0:
            return None

        return index - 1 if index > 0 else 0

    def updateSegmentWithLiveTracing(self, segmentId: SegmentId, roughTracing, updatedIdx, replaceLog: bool = False):
        """
        Updates a segment with live tracing.

        Args:
            segmentId (str): The ID of the segment.
        """

        if len(roughTracing) == 1:
            self.updateSegment(segmentId, Segment(
                roughTracing=Point(roughTracing[0]),
                segment=LineString([])
            ), replaceLog)
            return

        roughTracing = LineString(roughTracing)
        segment = self.segments[segmentId, "segment"]
        segment = segment if segment is not None and len(
            segment.coords) > 0 else None
        segment = self.optimizeSegment(roughTracing, segment, int(
            updatedIdx), live=True)
        update = Segment(roughTracing=roughTracing)

        if segment is not None:
            update.segment = segment

        self.updateSegment(segmentId, update, replaceLog)

    def onDelete(self):
        return False
