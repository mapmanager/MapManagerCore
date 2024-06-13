from dataclasses import dataclass
import warnings
from mapmanagercore.utils import force_2d
from ...layers.polygon import PolygonLayer
from ...config import Colors, Config, SegmentId, SpineId
from ...layers import LineLayer, PointLayer, Layer
from ...benchmark import timer
from shapely.geometry import Point, LineString
from shapely.errors import ShapelyDeprecationWarning
from .interactions import AnnotationsInteractions
from typing import List, Tuple
from typing import List
from ...layers.layer import Layer
from typing import List
import geopandas as gpd
from typing import TypedDict, Tuple


class AnnotationsSelection(TypedDict):
    """
    Represents a selection of annotations.

    Attributes:
      segmentID (str): The ID of the segment.
      spineID (str): The ID of the spine.
    """
    segmentID: SegmentId
    segmentIDEditing: SegmentId
    segmentIDEditingPath: SegmentId
    spineID: SpineId


class AnnotationsOptions(TypedDict):
    """
    Represents the options for annotations.

    Attributes:
      zRange (Tuple[int, int]): The visible z slice range.
      annotationSelections (AnnotationsSelection): The selected annotations.
      showLineSegments (bool): Flag indicating whether to show line segments.
      showLineSegmentsRadius (bool): Flag indicating whether to show line segment radius.
      showLabels (bool): Flag indicating whether to show labels.
      showAnchors (bool): Flag indicating whether to show anchors.
      showSpines (bool): Flag indicating whether to show spines.
      colorOn (str): The column to color the spines with.
      symbolOn (str): The column to use as the symbol of the spine.
    """
    zRange: Tuple[int, int]
    annotationSelections: AnnotationsSelection
    showLineSegments: bool
    showLineSegmentsRadius: bool
    showLabels: bool
    showAnchors: bool
    showSpines: bool

    colorOn: str
    symbolOn: str


@dataclass
class SegmentEditState:
    selectedIndex: int
    segmentId: SegmentId
    hoverSegment: LineString

    def __init__(self):
        self.selectedIndex = None
        self.segmentId = None
        self.hoverSegment = None

    def setSelectedIndex(self, index: int):
        if self.selectedIndex == index:
            return False
        self.selectedIndex = index
        return True

    def clear(self):
        self.selectedIndex = None
        self.hoverSegment = None


class AnnotationsLayers(AnnotationsInteractions):
    """Annotations Layers Generation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmentEditState = SegmentEditState()

    @timer
    def getAnnotations(self, options: AnnotationsOptions) -> list[Layer]:
        """
        Generates the annotations based on the provided options.

        Args:
            options (AnnotationsOptions): The options for retrieving annotations.

        Returns:
            list: A list of layers containing the retrieved annotations.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ShapelyDeprecationWarning)
            layers = []

            zRange = options["zRange"]
            selections = options["annotationSelections"]
            segmentIDEditingPath = selections["segmentIDEditingPath"]

            if segmentIDEditingPath:
                layers.extend(self._getEditingSegment(
                    zRange,
                    segmentIDEditingPath))
            else:
                self.segmentEditState.clear()
                if options["showLineSegments"]:
                    layers.extend(self._getSegments(
                        zRange,
                        selections["segmentIDEditing"],
                        selections["segmentID"],
                        options["showLineSegmentsRadius"]))

                if options["showSpines"]:
                    layers.extend(self._getSpines(options))

            layers = [layer for layer in layers if not layer.empty()]

            return layers

    @timer
    def _getSpines(self, options: AnnotationsOptions) -> list[Layer]:
        zRange = options["zRange"]
        selections = options["annotationSelections"]
        selectedSpine = selections["spineID"]
        editingSegmentId = selections["segmentIDEditing"]
        editing = editingSegmentId is not None
        # index_filter = options["filters"]

        layers = []
        if editing:
            # only show selected points
            points = self.points[self.points["segmentID"] == editingSegmentId]
        else:
            points = self.points

        points = points[["point", "anchorLine", "anchor", "z", "anchorZ"]]

        visiblePoints = points["z"].between(
            zRange[0], zRange[1], inclusive="left")
        visibleAnchors = points["anchorZ"].between(
            zRange[0], zRange[1], inclusive="left")

        if not editing:
            points = points[visiblePoints | visibleAnchors]

        if points.index.empty:
            return layers

        colorOn = options["colorOn"] if "colorOn" in options else None
        colors = self.getColors(colorOn, function=True)

        spines = (PointLayer(points["point"])
                  .id("spine")
                  .on("select", "spineID")
                  .fill(lambda id: Colors.selectedSpine if id == selectedSpine else colors(id)))

        labels = None
        if options["showAnchors"] or options["showLabels"]:
            anchorLines = (LineLayer(points["anchorLine"])
                           .id("anchorLines")
                           .stroke(Colors.anchorLine))

            if options["showLabels"]:
                labels = (anchorLines
                          .copy(id="label")
                          .extend(Config.labelOffset)
                          .tail()
                          .label()
                          .fill(Colors.label))

            if options["showAnchors"]:
                layers.extend(anchorLines.splitGhost(
                    visiblePoints & visibleAnchors, opacity=Config.ghostOpacity))

                anchors = (PointLayer(points["anchor"]).id("anchor")
                           .fill(Colors.anchorPoint))
                if editing:
                    anchors = (anchors.onDrag(self.moveAnchor)
                               .radius(Config.pointRadiusEditing))
                else:
                    anchors = anchors.radius(Config.pointRadius)

                layers.extend(anchors.splitGhost(
                    visibleAnchors, opacity=Config.ghostOpacity))

        if editing:
            spines = (spines.onDrag(self.moveSpine)
                      .radius(Config.pointRadiusEditing))
        else:
            spines = spines.radius(Config.pointRadius)

        # partially show spines that are not in scope with anchors in scope
        layers.extend(spines.splitGhost(
            visiblePoints, opacity=Config.ghostOpacity))

        # render labels
        if options["showLabels"]:
            layers.extend(labels.splitGhost(
                visiblePoints, opacity=Config.ghostOpacity))

        if selectedSpine in self.points.index:
            self._appendRois(selectedSpine, editing, layers)

        return layers

    @timer
    def _appendRois(self, selectedSpine: SpineId, editing: bool, layers: List[Layer]):
        boarderWidth = Config.roiStrokeWidth
        points = self.points[[selectedSpine]]

        headLayer = (PolygonLayer(points.loc[[selectedSpine], "roiHead"])
                     .id("roi-head")
                     .strokeWidth(boarderWidth)
                     .stroke(Colors.roiHead))

        baseLayer = (PolygonLayer(points.loc[[selectedSpine], "roiBase"])
                     .id("roi-base")
                     .strokeWidth(boarderWidth)
                     .stroke(Colors.roiBase))

        backgroundRoiHead = (headLayer
                             .copy(id="background", series=points.loc[[selectedSpine], "roiHeadBg"])
                             .stroke(Colors.roiHeadBg))

        backgroundRoiBase = (baseLayer
                             .copy(id="background", series=points.loc[[selectedSpine], "roiBaseBg"])
                             .stroke(Colors.roiBaseBg))
        if editing:
            # Add larger interaction targets
            layers.append(backgroundRoiHead.copy(id="translate")
                          .stroke(Colors.transparent)
                          .fill(Colors.transparent)
                          .onDrag(self.moveBackgroundRoi))
            layers.append(backgroundRoiBase.copy(id="translate")
                          .fill(Colors.transparent)
                          .stroke(Colors.transparent)
                          .onDrag(self.moveBackgroundRoi))

        layers.append(backgroundRoiHead)
        layers.append(backgroundRoiBase)
        layers.append(headLayer)
        layers.append(baseLayer)

        if editing:
            # Add the extend interaction target
            line = LineLayer(points.loc[[selectedSpine], "anchorLine"])

            layers.append(line.copy()
                          .extend(points.loc[selectedSpine, "roiExtend"])
                          .tail()
                          .radius(1)
                          .id("translate-extend")
                          .fill([255, 255, 255])
                          .fixed()
                          .stroke(Colors.roiHead)
                          .strokeWidth(1)
                          .onDrag(self.moveRoiExtend))

            layers.append(line.copy()
                          .offset(-points.loc[selectedSpine, "roiRadius"])
                          .normalize()
                          .tail()
                          .radius(1)
                          .id("translate-radius-right")
                          .fill([255, 255, 255])
                          .fixed()
                          .stroke(Colors.roiHead)
                          .strokeWidth(1)
                          .onDrag(self.moveRoiRadius))

            layers.append(line
                          .offset(points.loc[selectedSpine, "roiRadius"])
                          .normalize()
                          .tail()
                          .radius(1)
                          .id("translate-radius-left")
                          .fill([255, 255, 255])
                          .fixed()
                          .stroke(Colors.roiHead)
                          .strokeWidth(1)
                          .onDrag(self.moveRoiRadius))

    @timer
    def _getEditingSegment(self, zRange: Tuple[int, int], editSegPathId: SegmentId) -> List[Layer]:
        segments = self.segments[editSegPathId, ["roughTracing", "segment"]]
        self.segmentEditState.segmentId = editSegPathId
        _, _, x, y = self.shape

        def onClickHitTarget(_, x, y, z):
            self.segmentEditState.hoverSegment = None
            index = self.appendSegmentPoint(editSegPathId, x, y, z, False)
            if index is None:
                return False
            self.segmentEditState.setSelectedIndex(index)
            return True

        def onHoverHitTarget(_, x, y, z):
            oldSegment = self.appendSegmentPoint(
                editSegPathId, x, y, z, True)

            if oldSegment is None and self.segmentEditState.hoverSegment is None:
                return False

            self.segmentEditState.hoverSegment = oldSegment
            return True

        def onHoverOutHitTarget():
            if self.segmentEditState.hoverSegment is None:
                return False
            self.segmentEditState.hoverSegment = None
            return True

        hitTarget = (
            PolygonLayer.box(0, 0, x, y)
            .id("hitTarget")
            .fill(Colors.transparent)
            .onClick(onClickHitTarget)
            .onHover(onHoverHitTarget)
            .onHoverOut(onHoverOutHitTarget)
        )

        segment = (LineLayer(segments["segment"])
                   .id("segment")
                   .clipZ(zRange)
                   .stroke(Colors.segment))

        segmentGhost = (LineLayer(force_2d(segments["segment"]))
                        .id("segment-ghost")
                        .stroke(Colors.segment)
                        .opacity(Config.ghostOpacity))

        def addPoint(segId, x, y, z):
            idx = self.injectSegmentPoint(segId, x, y, z)
            if idx is None:
                return False

            self.segmentEditState.setSelectedIndex(idx)
            return True

        segmentGhost2 = (segmentGhost.copy(id="adder")
                         .stroke(Colors.transparent)
                         .strokeWidth(4)
                         .opacity(0)
                         .onClick(addPoint))

        points = gpd.GeoSeries(segments["roughTracing"])
        points = points.get_coordinates(ignore_index=True, include_z=True)
        points = points.apply(lambda x: Point(x[0], x[1], x[2]), axis=1)

        def pointColor(sid):
            if sid == self.segmentEditState.selectedIndex:
                return Colors.segmentEditing
            return Colors.segment

        def onDrag(idx, x, y, z, dragState):
            self.segmentEditState.setSelectedIndex(idx)
            return self.moveSegmentPoint(editSegPathId, x, y, z, idx, dragState)

        pointsLayer, pointsLayer2 = (PointLayer(gpd.GeoSeries(points))
                                     .id("roughTracing-points")
                                     .radius(1)
                                     .fill(pointColor)
                                     .fixed()
                                     .strokeWidth(1)
                                     .stroke(pointColor)
                                     .onDrag(onDrag)
                                     .onClick(lambda idx, x, y, z: self.segmentEditState.setSelectedIndex(idx))
                                     .splitZ(zRange))

        pointsLayer2 = pointsLayer2.opacity(
            Config.ghostOpacity).id("roughTracing-points-ghost")

        layers = []
        if self.segmentEditState.hoverSegment:
            layers.append(
                LineLayer(force_2d(gpd.GeoSeries(
                    [self.segmentEditState.hoverSegment])))
                .id("hover-segment")
                .stroke(Colors.pendingSegment)
                .opacity(255 * 0.65))

        layers.extend([hitTarget,
                       segment,
                       segmentGhost,
                       segmentGhost2,
                       pointsLayer,
                       pointsLayer2
                       ])
        return layers

    @timer
    def _getSegments(self, zRange: Tuple[int, int], editSegId: SegmentId, selectedSegId: SegmentId, showLineSegmentsRadius: bool) -> List[Layer]:
        layers = []
        segments = self.segments[["segment", "radius"]]

        def getStrokeColor(id: SegmentId):
            return Colors.segmentEditing if id == editSegId else (Colors.segmentSelected if id == selectedSegId else Colors.segment)

        segment = (LineLayer(segments["segment"])
                   .id("segment")
                   .clipZ(zRange)
                   .on("select", "segmentID")
                   .on("edit", "segmentIDEditing")
                   .stroke(getStrokeColor))

        boarderWidth = Config.segmentLeftRightStrokeWidth

        def offset(id: int):
            return segments.loc[id, "radius"] / boarderWidth

        # Render the ghost of the edit
        if editSegId is not None:
            self._segmentGhost(segments.loc[[editSegId], "segment"], showLineSegmentsRadius,
                               layers, segment, boarderWidth, offset)

        if showLineSegmentsRadius:
            # Left line
            left = (segment.copy(id="left")
                    .strokeWidth(boarderWidth)
                    .offset(lambda id: -offset(id)))

            layers.append(left)

            # Right line
            right = (segment.copy(id="right")
                     .strokeWidth(boarderWidth)
                     .offset(offset))
            layers.append(right)

        if editSegId is None:
            # Make the click target larger
            layers.append(segment.copy(id="interaction")
                          .strokeWidth(lambda id: segments.loc[id, "radius"])
                          .stroke(Colors.transparent))
        else:
            segment = segment.on("edit", "segmentIDEditingPath")

        # Add the line segment
        layers.append(segment.strokeWidth(
            lambda id: Config.segmentBoldWidth if id == editSegId else Config.segmentWidth))

        return layers

    @timer
    def _segmentGhost(self, segmentSeries, showLineSegmentsRadius: bool, layers: List[Layer], segment: LineLayer, boarderWidth: int, offset):
        segmentSeries = force_2d(segmentSeries)
        # segmentSeries = force_2d(self.segments[segId, "segment"])
        ghost = (segment.copy(segmentSeries, id="ghost")
                 .opacity(Config.ghostOpacity))

        if showLineSegmentsRadius:
            # Ghost Left line
            left = (ghost.copy(id="left-ghost")
                    .strokeWidth(boarderWidth)
                    .offset(lambda id: -offset(id)))
            layers.append(left)

            # Ghost Right line
            right = (ghost.copy(id="right-ghost")
                          .strokeWidth(boarderWidth)
                          .offset(offset))
            layers.append(right)

            def offset4(id: int):
                return offset(id) / 4

            layers.append(
                left.copy(id="interaction")
                .strokeWidth(boarderWidth * 4)
                .offset(offset4)
                .stroke(Colors.transparent)
                .opacity(0.0)
                .onDrag(self.moveSegmentRadius))

            layers.append(right.copy(
                id="interaction")
                .offset(lambda id: -offset4(id))
                .strokeWidth(boarderWidth * 4).stroke(Colors.transparent)
                .opacity(0.0)
                .onDrag(self.moveSegmentRadius))

            # Add the ghost
        layers.append(ghost.on("edit", "segmentIDEditingPath"))

    def onDelete(self):
        if self.segmentEditState.selectedIndex is None:
            return super().onDelete()

        self.segmentEditState.selectedIndex = self.deleteSegmentPoint(
            self.segmentEditState.segmentId, self.segmentEditState.selectedIndex)

        self.segmentEditState.hoverSegment = None
        return True
