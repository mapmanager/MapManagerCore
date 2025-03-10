from typing import Literal, Optional
from .base import SingleTimePointAnnotationsBase


class AnnotationsUtils(SingleTimePointAnnotationsBase):
  def nextSpine(spineId: int, offset: Literal[1,-1]) -> Optional[int]:
    """
    Returns the next spine in the current time point by moving either forward or backward.
    if the offset is 1, it will return the next spine, if the offset is -1, it will return the previous spine.
    If there is no spine in the given direction, it will return None.
    """
    # TODO: implement
    return None