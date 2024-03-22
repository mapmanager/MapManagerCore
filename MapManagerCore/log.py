from __future__ import annotations
from enum import Enum
from typing import Generic, List, Union, TypeVar
import numpy as np
import pandas as pd
import geopandas as gp

T = TypeVar('T')


class Op(Generic[T]):
    type: T
    deleted: pd.DataFrame
    added: pd.DataFrame
    changed: pd.DataFrame

    def __init__(self, type: T, before: gp.GeoDataFrame, after: gp.GeoDataFrame):
        self.type = type
        commonIndexes = before.index.intersection(after.index)

        self.changed = gp.GeoDataFrame(before.loc[commonIndexes]).compare(
            gp.GeoDataFrame(after.loc[commonIndexes]), result_names=("before", "after"))

        self.deleted = before.loc[before.index.difference(
            commonIndexes).values]
        self.added = after.loc[after.index.difference(commonIndexes).values]

    def isEmpty(self) -> bool:
        return self.deleted.empty and self.added.empty and self.changed.empty

    def update(self, operation: Op) -> bool:
        if self.type != operation.type:
            return False
        if self.changed.index != operation.changed.index:
            return False
        if self.deleted.index != operation.deleted.index:
            return False
        if self.added.index != operation.added.index:
            return False

        for key, state in operation.changed.columns.values:
            if state == "after":
                self.changed.loc[operation.changed.index, (key, "after")] = operation.changed[(key, "after")]

        return True

    def reverse(self, df: gp.GeoDataFrame):
        df.drop(self.added.index, inplace=True)
        for key, operation in self.changed.columns:
            if operation == "before":
                df.loc[self.changed.index, key] = self.changed[(key, "before")]

        for key, values in self.deleted.iterrows():
            df.loc[key] = values

        now = np.datetime64("now")
        df.loc[self.changed.index.union(self.deleted.index).values,
               "modified"] = now

    def apply(self, df: gp.GeoDataFrame):
        df.drop(self.deleted.index, inplace=True)
        for key, operation in self.changed.columns.values:
            if operation == "after":
                df.loc[self.changed.index, key] = self.changed[(key, "after")]

        for key, values in self.added.iterrows():
            df.loc[key] = values

        now = np.datetime64("now")
        df.loc[self.changed.index.union(self.added.index).values,
               "modified"] = now


class RecordLog(Generic[T]):
    operations: List[Op[T]]

    def __init__(self):
        self.operations = []
        self.index = -1
        self.replaceable = False
        return

    def createState(self):
        self.replaceable = False

    def _peakReplaceable(self) -> Union[Op[T], None]:
        """
        Peeks the next operation to be undone if replaceable.

        Returns:
            Union[Op[T], None]: The next operation to be undone, or None if there are no more operations to undo.
        """
        if self.index < 0 or not self.replaceable:
            return None

        self.operations = self.operations[:self.index + 1]
        return self.operations[self.index]

    def push(self, operation: Op[T], replace=False):
        """
        Pushes an operation to the log.

        Args:
            operation (T): The operation to be pushed to the log.
        """

        if replace:
            # replace the last operation in the log
            peak = self._peakReplaceable()
            if peak is not None and peak.update(operation):
                return

        if self.index < len(self.operations) - 1:
            self.operations = self.operations[:self.index + 1]

        self.operations.append(operation)
        self.index += 1
        self.replaceable = True

    def undo(self) -> Union[Op[T], None]:
        """
        Undoes the last operation in the log.

        Returns:
            Union[Op[T], None]: The undone operation, or None if there are no more operations to undo.
        """
        self.replaceable = False
        if self.index < 0:
            return None

        operation = self.operations[self.index]
        self.index -= 1
        return operation

    def redo(self) -> Union[Op[T], None]:
        """
        Redoes the last undone operation in the log.

        Returns:
            Union[Op[T], None]: The redone operation, or None if there are no more operations to redo.
        """
        self.replaceable = False
        if self.index >= len(self.operations) - 1:
            return None

        self.index += 1
        operation = self.operations[self.index]
        return operation
