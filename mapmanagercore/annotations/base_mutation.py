import datetime
from typing import Tuple, Union
import geopandas as gp
import numpy as np
import pandas as pd
from mapmanagercore.types import SpineId
from mapmanagercore.config import LineSegment, Spine
from mapmanagercore.loader.base import Loader
from mapmanagercore.utils import validateColumns
from ..log import Op, RecordLog
from enum import Enum
from .base import AnnotationsBase

Key = Union[SpineId, Tuple[SpineId, int]]
Keys = Union[Key, list[Key]]


class AnnotationType(Enum):
    Point = 1
    LineSegment = 2


class AnnotationsBaseMut(AnnotationsBase):
    _log: RecordLog[AnnotationType]

    def __init__(self, loader: Loader):
        super().__init__(loader)
        self._log = RecordLog()

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

    def deleteSpine(self, spineId: Keys, skipLog=False) -> None:
        self._delete(AnnotationType.Point, spineId, skipLog=skipLog)

    def deleteSegment(self, segmentId: Keys, skipLog=False) -> None:
        self._delete(AnnotationType.LineSegment, segmentId, skipLog=skipLog)

    def _delete(self, type: AnnotationType, id: Keys, skipLog=False) -> None:
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

    def updateSpine(self, spineId: Keys, value: Spine, replaceLog=False, skipLog=False):
        """
        Set the spine with the given ID to the specified value.

        Parameters:
            spineId (str): The ID of the spine.
            value (Union[dict, gp.Series, pd.Series]): The value to set for the spine.
        """
        # if not spineId in self._points.index:
        #     raise ValueError(f"Spine with ID {spineId} not found")
        validateColumns(value, Spine)
        
        if "t" in value:
            raise ValueError(
                f"Invalid type for column 't' must be set on the spine key")

        if "spineID" in value:
            raise ValueError(
                f"Invalid type for column 'spineID' must be set on the spine key")

        return self._update(AnnotationType.Point, spineId, value, replaceLog, skipLog)

    def updateSegment(self, segmentId: Keys, value: LineSegment, replaceLog=False, skipLog=False):
        """
        Set the segment with the given ID to the specified value.

        Parameters:
            segmentId (str): The ID of the spine.
            value (Union[dict, gp.Series, pd.Series]): The value to set for the spine.
        """
        # if not segmentId in self._lineSegments.index:
        #     raise ValueError(f"Segment with ID {segmentId} not found")
        validateColumns(value, LineSegment)
        
        if "t" in value:
            raise ValueError(
                f"Invalid type for column 't' must be set on the segment key")

        if "segmentID" in value:
            raise ValueError(
                f"Invalid type for column 'segmentID' must be set on the segment key")

        return self._update(AnnotationType.LineSegment, segmentId, value, replaceLog, skipLog)

    def _update(self, type: AnnotationType, ids: Keys, value: Union[dict, gp.GeoSeries, pd.Series], replaceLog=False, skipLog=False):
        df = self._getDf(type)
        ids = [ids] if isinstance(ids, tuple) or isinstance(ids, int) else ids

        old = df.loc[df.index.intersection(ids) if isinstance(
            ids[0], tuple) else df.index.get_level_values(0).intersection(ids)].copy()
        value = pd.Series(value)

        updateDataFrame(df, ids, value)

        op = Op(type, old, df.loc[ids])
        df.loc[ids, "modified"] = np.datetime64(datetime.datetime.now())
        df.sort_index(inplace=True)

        if skipLog:
            return

        # Add the operation to the log
        if op.isEmpty():
            if replaceLog:
                return

            self._log.createState()
            return

        self._log.push(op, replace=replaceLog)


def updateDataFrame(df: gp.GeoDataFrame, ids: list[Key], value: pd.Series):
    for (i, id) in enumerate(ids):
        oldId = id

        # Update indexes when ids change
        if isinstance(id, tuple):
            id = list(id)
            id_changed = False
            for idx, name in enumerate(df.index.names):
                if name in value.index:
                    id[idx] = value[name]
                    id_changed = True
            id = tuple(id)

            if id_changed:
                ids[i] = id
                df.loc[id, :] = df.loc[oldId, :]
                df.drop(oldId, inplace=True)

        else:
            name = df.index.names[0]
            if name in value.index:
                ids[i] = id = value[name]
                df.rename(index={oldId: id}, inplace=True)

        df.loc[id, value.index.values] = value.values
