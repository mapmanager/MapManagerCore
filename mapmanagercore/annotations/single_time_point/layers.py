import warnings

from ...layers.polygon import PolygonLayer
from ...config import COLORS, CONFIG, TRANSPARENT, SegmentId, SpineId, scaleColors, symbols
from ...layers import LineLayer, PointLayer, Layer
from ...layers.utils import dropZ
from ...benchmark import timer
import warnings
from shapely.errors import ShapelyDeprecationWarning
from .interactions import AnnotationsInteractions
from typing import List, Tuple
from typing import List
from ...layers.layer import Layer
from typing import List
import pandas as pd
from typing import TypedDict, Tuple
from plotly.express.colors import sample_colorscale


class AnnotationsSelection(TypedDict):
    """
    Represents a selection of annotations.

    Attributes:
      segmentID (str): The ID of the segment.
      spineID (str): The ID of the spine.
    """
    segmentID: SegmentId
    segmentIDEditing: SegmentId
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


class AnnotationsLayers(AnnotationsInteractions):
    """Annotations Layers Generation"""

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
            points = self[self._points["segmentID"] == editingSegmentId]
        else:
            points = self

        visiblePoints = points["z"].between(
            zRange[0], zRange[1], inclusive="left")
        visibleAnchors = points["anchorZ"].between(
            zRange[0], zRange[1], inclusive="left")

        if not editing:
            points = points[visiblePoints | visibleAnchors]

        if len(points._points) == 0:
            return layers

        colorOn = options["colorOn"] if "colorOn" in options else None
        colors = points.getColors(colorOn, function=True)

        spines = (PointLayer(points["point"])
                  .id("spine")
                  .on("select", "spineID")
                  .fill(lambda id: COLORS["selectedSpine"] if id == selectedSpine else colors(id)))

        labels = None
        if options["showAnchors"] or options["showLabels"]:
            anchorLines = (spines
                           .copy(id="anchorLine")
                           .toLine(points["anchor"])
                           .stroke(COLORS["anchorLine"]))

            if options["showLabels"]:
                labels = (anchorLines
                          .copy(id="label")
                          .extend(CONFIG["labelOffset"])
                          .tail()
                          .label()
                          .fill(COLORS["label"]))

            if options["showAnchors"]:
                layers.extend(anchorLines.splitGhost(
                    visiblePoints & visibleAnchors, opacity=CONFIG["ghostOpacity"]))

                anchors = (PointLayer(points["anchor"]).id("anchor")
                           .fill(COLORS["anchorPoint"]))
                if editing:
                    anchors = (anchors.onDrag(self.moveAnchor)
                               .radius(CONFIG["pointRadiusEditing"]))
                else:
                    anchors = anchors.radius(CONFIG["pointRadius"])

                layers.extend(anchors.splitGhost(
                    visibleAnchors, opacity=CONFIG["ghostOpacity"]))

        if editing:
            spines = (spines.onDrag(self.moveSpine)
                      .radius(CONFIG["pointRadiusEditing"]))
        else:
            spines = spines.radius(CONFIG["pointRadius"])

        # partially show spines that are not in scope with anchors in scope
        layers.extend(spines.splitGhost(
            visiblePoints, opacity=CONFIG["ghostOpacity"]))

        # render labels
        if options["showLabels"]:
            layers.extend(labels.splitGhost(
                visiblePoints, opacity=CONFIG["ghostOpacity"]))

        if selectedSpine in self._points.index:
            self._appendRois(selectedSpine, editing, layers)

        return layers

    @timer
    def _appendRois(self, selectedSpine: SpineId, editing: bool, layers: List[Layer]):
        boarderWidth = CONFIG["roiStrokeWidth"]

        headLayer = (PolygonLayer(self[[selectedSpine], "roiHead"])
                     .id("roi-head")
                     .strokeWidth(boarderWidth)
                     .stroke(COLORS["roiHead"]))

        baseLayer = (PolygonLayer(self[[selectedSpine], "roiBase"])
                     .id("roi-base")
                     .strokeWidth(boarderWidth)
                     .stroke(COLORS["roiBase"]))

        backgroundRoiHead = (headLayer
                             .copy(id="background", series=self[[selectedSpine], "roiHeadBg"])
                             .stroke(COLORS["roiHeadBg"]))

        backgroundRoiBase = (baseLayer
                             .copy(id="background", series=self[[selectedSpine], "roiBaseBg"])
                             .stroke(COLORS["roiBaseBg"]))
        if editing:
            # Add larger interaction targets
            layers.append(backgroundRoiHead.copy(id="translate")
                          .stroke(TRANSPARENT)
                          .fill(TRANSPARENT)
                          .onDrag(self.moveBackgroundRoi))
            layers.append(backgroundRoiBase.copy(id="translate")
                          .fill(TRANSPARENT)
                          .stroke(TRANSPARENT)
                          .onDrag(self.moveBackgroundRoi))

        layers.append(backgroundRoiHead)
        layers.append(backgroundRoiBase)
        layers.append(headLayer)
        layers.append(baseLayer)

        if editing:
            # Add the extend interaction target
            line = PointLayer(self[[selectedSpine], "point"]).toLine(
                self[[selectedSpine], "anchor"])

            layers.append(line.copy()
                          .extend(self[selectedSpine, "roiExtend"])
                          .tail()
                          .radius(1)
                          .id("translate-extend")
                          .fill([255, 255, 255])
                          .fixed()
                          .stroke(COLORS["roiHead"])
                          .strokeWidth(1)
                          .onDrag(self.moveRoiExtend))

            layers.append(line.copy()
                          .offset(-self[selectedSpine, "roiRadius"])
                          .normalize()
                          .tail()
                          .radius(1)
                          .id("translate-radius-right")
                          .fill([255, 255, 255])
                          .fixed()
                          .stroke(COLORS["roiHead"])
                          .strokeWidth(1)
                          .onDrag(self.moveRoiRadius))

            layers.append(line
                          .offset(self[selectedSpine, "roiRadius"])
                          .normalize()
                          .head()
                          .radius(1)
                          .id("translate-radius-left")
                          .fill([255, 255, 255])
                          .fixed()
                          .stroke(COLORS["roiHead"])
                          .strokeWidth(1)
                          .onDrag(self.moveRoiRadius))

    @timer
    def _getSegments(self, zRange: Tuple[int, int], editSegId: SegmentId, selectedSegId: SegmentId, showLineSegmentsRadius: bool) -> List[Layer]:
        layers = []
        segments = self.segments

        def getStrokeColor(id: SegmentId):
            return COLORS["segmentEditing"] if id == editSegId else (COLORS["segmentSelected"] if id == selectedSegId else COLORS["segment"])

        segment = (LineLayer(segments["segment"])
                   .id("segment")
                   .clipZ(zRange)
                   .on("select", "segmentID")
                   .on("edit", "segmentIDEditing")
                   .stroke(getStrokeColor))

        boarderWidth = CONFIG["segmentLeftRightStrokeWidth"]

        def offset(id: int):
            return self._lineSegments.loc[id, "radius"] / boarderWidth

        # Render the ghost of the edit
        if editSegId is not None:
            self._segmentGhost(editSegId, showLineSegmentsRadius,
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
                          .strokeWidth(lambda id: self._lineSegments.loc[id, "radius"])
                          .stroke(TRANSPARENT))

        # Add the line segment
        layers.append(segment.strokeWidth(
            lambda id: CONFIG["segmentBoldWidth"] if id == editSegId else CONFIG["segmentWidth"]))

        return layers

    def _segmentGhost(self, segId: SegmentId, showLineSegmentsRadius: bool, layers: List[Layer], segment: LineLayer, boarderWidth: int, offset):
        segmentSeries = self.segments[segId, "segment"].apply(dropZ)
        ghost = (segment.copy(segmentSeries, id="ghost")
                 .opacity(CONFIG["ghostOpacity"]))

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
                .stroke(TRANSPARENT)
                .opacity(0.0)
                .onDrag(self.moveSegmentRadius))

            layers.append(right.copy(
                id="interaction")
                .offset(lambda id: -offset4(id))
                .strokeWidth(boarderWidth * 4).stroke(TRANSPARENT)
                .opacity(0.0)
                .onDrag(self.moveSegmentRadius))

            # Add the ghost
        layers.append(ghost)

    def getColors(self, colorOn: str = None, function=False) -> pd.Series:
        if colorOn is None:
            if function:
                return lambda _: COLORS["spine"]
            return pd.Series([COLORS["spine"]] * len(self._points), index=self.index)

        categorical = False
        if colorOn not in self.columnsAttributes:
            raise ValueError(f"Column {colorOn} has no color attributes.")

        attr = self.columnsAttributes[colorOn]
        if "colors" in attr:
            colors = attr["colors"]
        elif "categorical" in attr and attr["categorical"]:
            colors = COLORS["categorical"]
            categorical = True
        elif "divergent" in attr and attr["divergent"]:
            colors = COLORS["divergent"]
        else:
            colors = COLORS["scalar"]

        values = self[colorOn]
        if categorical and not isinstance(colors, dict):
            keys = list(values.unique())
            keys.sort()
            originalColors = colors
            colors = {key: originalColors[i % len(
                originalColors)] for i, key in enumerate(keys)}

        if isinstance(colors, dict):
            if function:
                return lambda x: colors[x]
            return values.apply(lambda x: colors[values[x]])

        valuesMin = values.min()
        valuesMax = values.max()

        colors = scaleColors(colors, 1.0/255.0)
        if function:
            return lambda x: scaleColors(sample_colorscale(colors, (values[x]-valuesMin)/(valuesMax-valuesMin), colortype="tuple"), 255)

        normalized = (values-valuesMin)/(valuesMax-valuesMin)
        return pd.Series(scaleColors(sample_colorscale(colors, normalized, colortype="tuple"), 255), index=values.index)

    def getSymbols(self, shapeOn: str, function=False) -> pd.Series:
        if shapeOn not in self.columnsAttributes:
            raise ValueError(f"Column {shapeOn} has no shape attributes.")

        attr = self.columnsAttributes[shapeOn]
        if "symbols" in attr:
            symbols_ = attr["symbols"]
        elif "categorical" in attr and attr["categorical"]:
            symbols_ = symbols
        else:
            raise ValueError(
                f"Column {shapeOn} is scalar and cannot be used as a shape.")

        values = self[shapeOn]
        if not isinstance(symbols_, dict):
            keys = list(values.unique())
            keys.sort()
            originalSymbols = symbols_
            symbols_ = {key: originalSymbols[i % len(
                originalSymbols)] for i, key in enumerate(keys)}

        if function:
            return lambda x: symbols_[x]

        return values.apply(lambda x: symbols_[x])
