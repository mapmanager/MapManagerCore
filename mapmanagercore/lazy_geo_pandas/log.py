from __future__ import annotations
import datetime
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

        if isinstance(before, gp.GeoSeries) or isinstance(before, pd.Series):
            before = before.to_frame().T
        if isinstance(after, gp.GeoSeries) or isinstance(after, pd.Series):
            after = after.to_frame().T

        if commonIndexes.empty:
            self.deleted = before
            self.added = after
            self.changed = gp.GeoDataFrame()
            return

        self.changed = gp.GeoDataFrame(before.loc[commonIndexes]).compare(
            gp.GeoDataFrame(after.loc[commonIndexes]), result_names=("before", "after"))
        self.deleted = before.loc[before.index.difference(
            commonIndexes).values]
        self.added = after.loc[after.index.difference(commonIndexes).values]

    def isEmpty(self) -> bool:
        """
        Checks if the operation is empty (no op).
        """
        return self.deleted.empty and self.added.empty and self.changed.empty

    def update(self, operation: Op) -> bool:
        """
        Updates the operation by merging the new operation.

        Args:
            operation (Op): The new operation to be merged.

        Returns:
            bool: True if the operation has been merged, False otherwise.
        """
        if self.type != operation.type:
            return False

        # support modifying added values
        if self.changed.empty and not operation.changed.empty and self.deleted.empty and operation.added.empty and operation.deleted.empty:
            if self.added.index != operation.changed.index:
                return False
            for key, state in operation.changed.columns.values:
                if state == "after":
                    self.added.loc[operation.changed.index,
                                   key] = operation.changed[(key, "after")]
            return True

        if self.changed.index != operation.changed.index:
            return False
        if self.deleted.index != operation.deleted.index:
            return False
        if self.added.index != operation.added.index:
            return False

        for key, state in operation.changed.columns.values:
            if state == "after":
                self.changed.loc[operation.changed.index,
                                 (key, "after")] = operation.changed[(key, "after")]

        return True

    def reverse(self, df: gp.GeoDataFrame):
        """
        Reverses the state change onto the dataframe. (undo)
        """
        for key, operation in self.changed.columns:
            if operation == "before":
                df.loc[self.changed.index, key] = self.changed[(key, "before")]

        df.drop(self.added.index, inplace=True)

        for key, values in self.deleted.iterrows():
            df.loc[key, :] = values

        now = np.datetime64(datetime.datetime.now())
        changedIndex = self.changed.index.union(self.deleted.index).values
        df.loc[changedIndex, "modified"] = now

    def apply(self, df: gp.GeoDataFrame):
        """
        Applies the state change onto the dataframe. (redo)
        """
        for key, values in self.added.iterrows():
            df.loc[key, :] = values

        for key, operation in self.changed.columns.values:
            if operation == "after":
                df.loc[self.changed.index, key] = self.changed[(key, "after")]

        df.drop(self.deleted.index, inplace=True)

        now = np.datetime64(datetime.datetime.now())
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
        """
        Creates a new state on the log so future states will not replace the last state.
        """
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
