from typing import Union
import geopandas as gp
import numpy as np
import pandas as pd

from MapManagerCore.config import LineSegment, Spine
from MapManagerCore.loader.base import Loader
from MapManagerCore.utils import validateColumns
from ...log import Op, RecordLog
from enum import Enum
from .base import AnnotationsBase


class AnnotationType(Enum):
    Point = 1
    LineSegment = 2


class AnnotationsBaseMut(AnnotationsBase):
    _log: RecordLog[AnnotationType]

    def __init__(self, loader: Loader):
        super().__init__(loader)
        self._log = RecordLog()

    #  TODO: create from log: withLog(self, loader: ImageLoader, log: RecordLog):

    def undo(self):
        """
        Undo the last operation.
        """
        op = self._log.undo()
        if op is None:
            return

        op.reverse(self._getDf(op.type))

    def redo(self):
        """
        Redo the last undone operation.
        """
        op = self._log.redo()
        if op is None:
            return

        op.apply(self._getDf(op.type))

    def _getDf(self, type: AnnotationType) -> gp.GeoDataFrame:
        return {
            AnnotationType.Point: self._points,
            AnnotationType.LineSegment: self._lineSegments
        }[type]

    def deleteSpine(self, spineId: Union[str, list[str]], skipLog=False) -> None:
        self._delete(AnnotationType.Point, spineId, skipLog=skipLog)

    def deleteSegment(self, segmentId: Union[str, list[str]], skipLog=False) -> None:
        self._delete(AnnotationType.LineSegment, segmentId, skipLog=skipLog)

    def _delete(self, type: AnnotationType, id: Union[str, list[str]], skipLog=False) -> None:
        """
        Deletes a spine or segment.

        Args:
          Id (str): The ID of the spine or segment.
        """
        df = self._getDf(type)
        id = [id] if isinstance(id, str) else id
        if not skipLog:
            self._log.push(
                Op(type, df.loc[id], gp.GeoDataFrame(columns=df.columns)))

        df.drop(id, inplace=True)

    def updateSpine(self, spineId: Union[str, list[str]], value: Spine, replaceLog=False, skipLog=False):
        """
        Set the spine with the given ID to the specified value.

        Parameters:
            spineId (str): The ID of the spine.
            value (Union[dict, gp.Series, pd.Series]): The value to set for the spine.
        """
        validateColumns(value, Spine)
        return self._update(AnnotationType.Point, spineId, value, replaceLog, skipLog)

    def updateSegment(self, segmentId: Union[str, list[str]], value: LineSegment, replaceLog=False, skipLog=False):
        """
        Set the segment with the given ID to the specified value.

        Parameters:
            segmentId (str): The ID of the spine.
            value (Union[dict, gp.Series, pd.Series]): The value to set for the spine.
        """
        validateColumns(value, LineSegment)
        return self._update(AnnotationType.LineSegment, segmentId, value, replaceLog, skipLog)

    def _update(self, type: AnnotationType, ids: Union[str, list[str]], value: Union[dict, gp.GeoSeries, pd.Series], replaceLog=False, skipLog=False):
        df = self._getDf(type)
        ids = [ids] if isinstance(ids, str) else ids

        old = df.loc[df.index.intersection(ids)].copy()
        value = pd.Series(value)
        for id in ids:
            df.loc[id, value.index] = value

        op = Op(type, old, df.loc[ids])
        df.loc[ids, "modified"] = np.datetime64("now")

        if skipLog:
            return

        # Add the operation to the log
        if op.isEmpty():
            if replaceLog:
                return

            self._log.createState()
            return

        self._log.push(op, replace=replaceLog)
