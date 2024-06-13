from typing import Tuple, Union
from ..schemas import Spine, Segment
from ..config import SegmentId, SpineId
from .base import AnnotationsBase

from mapmanagercore.logger import logger

Key = Union[SpineId, Tuple[SpineId, int]]
Keys = Union[Key, list[Key]]


class AnnotationsBaseMut(AnnotationsBase):
    def deleteSpine(self, spineId: Keys, skipLog=False):
        """
        Delete the spine with the given ID.
        """
        self._drop("Spine", spineId, skipLog=skipLog)

    def deleteSegment(self, segmentId: Keys, skipLog=False):
        """
        Delete the segment with the given ID.
        """
        try:
            if not self.points[["segmentID"]].reset_index().set_index(["segmentID", "t"]).loc[segmentId].empty:
                # abb
                logger.info(f'Cannot delete segment(s) {segmentId} as it has an attached spine(s)')
                return False
                # raise ValueError(
                #     f"Cannot delete segment(s) {segmentId} as it has an attached spine(s)")
        except KeyError:
            pass

        self._drop("Segment", segmentId, skipLog=skipLog)
        return True
    
    def updateSpine(self, spineId: Keys, value: Spine, replaceLog=False, skipLog=False):
        """
        Set the spine with the given ID to the specified value.            
        """
        return self._update("Spine", spineId, value, replaceLog, skipLog)

    def updateSegment(self, segmentId: Keys, value: Segment, replaceLog=False, skipLog=False):
        """Set the segment with the given ID to the specified value.

        Args:
            segmentId (str): The ID of the spine.
            value (Union[dict, gp.Series, pd.Series]): The value to set for the spine.
        """
        return self._update("Segment", segmentId, value, replaceLog, skipLog)

    def newUnassignedSpineId(self) -> SpineId:
        """
        Returns a new unassigned spine ID.
        """
        if len(self.points) == 0:
            return 0
        return self.points.index.get_level_values(0).max() + 1

    def newUnassignedSegmentId(self) -> SegmentId:
        """
        Returns a new unassigned segment ID.
        """
        if len(self.segments) == 0:
            return 0
        return self.segments.index.get_level_values(0).max() + 1

    def connect(self, spineKey: Tuple[SpineId, int], toSpineKey: Tuple[SpineId, int]):
        
        # ValueError: Can only compare identically-labeled Series objects
        # if self.points[toSpineKey, "segmentID"] != self.points[spineKey, "segmentID"]:
        _segmentID = self.points[spineKey, "segmentID"].values[0]
        _toSegmentID = self.points[toSpineKey, "segmentID"].values[0]
        if _toSegmentID != _segmentID:
            raise ValueError("Cannot connect spines from different segments.")
        
        # check if the key already exists in the time point
        existingKey = (toSpineKey[0], spineKey[1])
        if existingKey in self.points.index:
            self.disconnect(existingKey)

        # Propagate the spine ID to all future time points
        self.updateSpine(slice(spineKey, spineKey[0]), Spine(
            spineID=toSpineKey[0],
        ))

    def disconnect(self, spineKey: Tuple[SpineId, int]):
        newID = self.newUnassignedSpineId()

        # Propagate the spine ID change to all future time points
        self.updateSpine(slice(spineKey, spineKey[0]), Spine(
            spineID=newID,
        ))

    def connectSegment(self, segmentKey: Tuple[SegmentId, int], toSegmentKey: Tuple[SegmentId, int]):
        if segmentKey[1] == toSegmentKey[1]:
            raise ValueError(
                "Cannot connect segments in the same time points.")

        # check if the key already exists in the time point
        existingKey = (toSegmentKey[0], segmentKey[1])
        if existingKey in self.segments.index:
            self.disconnectSegment(existingKey)

        # Propagate the segment ID to all future time points
        self.updateSegment(slice(segmentKey, segmentKey[0]), Segment(
            segmentID=toSegmentKey[0],
        ))

    def disconnectSegment(self, segmentKey: Tuple[SegmentId, int]):
        newID = self.newUnassignedSegmentId()

        # Propagate the segment ID change to all future time points
        self.updateSegment(slice(segmentKey, segmentKey[0]), Segment(
            segmentID=newID,
        ))
