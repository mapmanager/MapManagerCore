from typing import TypedDict, Tuple

SpineId = int
SegmentId = int


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


class ImageViewSelection(TypedDict):
    """
    Represents the image view state.

    Attributes:
      t (int): the time slot index.
      z (Tuple[int, int]): The visible z slice range.
    """
    t: int
    z: Tuple[int, int]


class AnnotationsOptions(TypedDict):
    """
    Represents the options for annotations.

    Attributes:
      selection (ImageViewSelection): The image view state.
      annotationSelections (AnnotationsSelection): The selected annotations.
      showLineSegments (bool): Flag indicating whether to show line segments.
      showLineSegmentsRadius (bool): Flag indicating whether to show line segment radius.
      showLabels (bool): Flag indicating whether to show labels.
      showAnchors (bool): Flag indicating whether to show anchors.
      showSpines (bool): Flag indicating whether to show spines.
      colorOn (str): The column to color the spines with.
      shapeOn (str): The column to shape the spines with.
    """
    selection: ImageViewSelection
    annotationSelections: AnnotationsSelection
    showLineSegments: bool
    showLineSegmentsRadius: bool
    showLabels: bool
    showAnchors: bool
    showSpines: bool

    colorOn: str
    shapeOn: str
