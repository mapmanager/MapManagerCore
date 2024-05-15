from copy import copy
import datetime
import io
from typing import Callable, Dict, Generic, Hashable, Iterator, List, Self, Set, TypeVar, Union
import numpy as np
import pandas as pd
from .attributes import _ColumnAttributes, ColumnAttributes
from .utils import updateDataFrame
from .schema import MISSING_VALUE, Schema
from .log import Op, RecordLog
import geopandas as gp
from collections.abc import Sequence


class LazyGeoPandas:
    _log: RecordLog[str]
    # keys are pre suffixed with .valid
    _dependents: dict[str, dict[str, dict[str, set[str]]]]

    @classmethod
    def setDefaultStore(cls, store):
        global SOURCE
        SOURCE = store

    def __init__(self) -> None:
        self._log = RecordLog()
        self._frames: dict[str, LazyGeoFrame] = {}
        self._dependents = {}

    def addSchema(self, frame) -> None:
        frame: LazyGeoFrame = frame
        frame._store = self
        key = frame._schema._key
        self._frames[key] = frame
        self.updateDependents(frame)

    def updateDependents(self, frame):
        frame: LazyGeoFrame = frame
        key = frame._schema._key
        # When a key is updated, we need to invalidate all the dependent keys
        # Here we pre compute the invalidation dependencies keys
        for column, attribute in frame._schema._attributes.items():
            if not "_dependencies" in attribute:
                continue

            for storeKey, deps in attribute["_dependencies"].items():
                if not storeKey in self._dependents:
                    self._dependents[storeKey] = {}
                storeDependents = self._dependents[storeKey]
                for dep in deps:
                    if not dep in storeDependents:
                        storeDependents[dep] = {}
                    depDependents = storeDependents[dep]

                    if not key in depDependents:
                        depDependents[key] = set()
                    depDependents[key].add(column + ".valid")

        changed = True
        if not key in self._dependents:
            return

        storeDependents = self._dependents[key]

        while changed:
            changed = False
            for key, attributes in frame._schema._attributes.items():
                if not "_func" in attributes:
                    continue
                if not key in storeDependents:
                    continue

                for storeKey, deps in attributes["_dependencies"].items():
                    for dep in deps:
                        didChanged = mergeDeps(
                            self._dependents[storeKey][dep], storeDependents[key])
                        changed = changed or didChanged

    def getFrame(self, key: str):
        return self._frames[key]

    def _getDf(self, key: str) -> gp.GeoDataFrame:
        return self._frames[key]._df

    def _drop(self, key: str, id: Union[Hashable, Sequence[Hashable], pd.Index], skipLog=False) -> None:
        store = self._frames[key]
        df = store._rootDf

        if not skipLog:
            ids = ids if isinstance(ids, pd.Index) or isinstance(
                ids, Sequence) else [ids]
            deletedData = df.loc[ids]
            self._log.push(
                Op(key, deletedData, gp.GeoDataFrame(columns=df.columns)))

        df.drop(id, inplace=True)
        store._state.increment()

    def _invalidateCached(self, ids: pd.Index, key: str, columns: Iterator[str]):
        store = self._frames[key]
        invalid = store._getDependentColumns(columns)
        for depKey, invalidateCols in invalid.items():
            depStore = self._frames[depKey]
            df = depStore._df
            columns = df.columns.intersection(invalidateCols)
            if depKey == key:
                df.loc[ids, columns] = False
                continue

            newIds = depStore._schema._reverseMapIds(key, df, store._df, ids)
            df.loc[newIds, columns] = False

        store._state.increment()

    def _update(self, key: str, ids: Union[Hashable, Sequence[Hashable], pd.Index], value: Schema, replaceLog=False, skipLog=False):
        store = self._frames[key]

        if not isinstance(value, store._schema):
            raise ValueError("Invalid value type", type(value))

        value = {key: val for key, val in vars(
            value).items() if val is not MISSING_VALUE}

        store._schema.validateColumns(value, dropIndex=True)

        ids = ids if isinstance(ids, pd.Index) or isinstance(
            ids, list) else [ids]

        df = store._df
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], tuple):
            old = df.loc[df.index.intersection(ids)].copy()
        else:
            old = df.loc[df.index.get_level_values(0).intersection(ids)].copy()

        value = pd.Series(value)

        updateDataFrame(df, ids, value)

        op = Op(key, old, df.loc[ids])
        df.loc[ids, "modified"] = np.datetime64(datetime.datetime.now())
        df.sort_index(inplace=True)
        if not op.isEmpty():
            changed = op.changed.columns.get_level_values(0).unique()
            self._invalidateCached(ids, key, changed.values)
        store._state.increment()

        if skipLog:
            return

        # Add the operation to the log
        if op.isEmpty():
            if replaceLog:
                return

            self._log.createState()
            return

        self._log.push(op, replace=replaceLog)

    def undo(self) -> None:
        op = self._log.undo()
        if op is None:
            return
        store = self.getFrame(op.type)
        op.reverse(store._rootDf)
        self._invalidateLogOpChanges(op)

    def redo(self) -> None:
        op = self._log.redo()
        if op is None:
            return
        store = self.getFrame(op.type)
        op.apply(store._rootDf)
        self._invalidateLogOpChanges(op)

    def _invalidateLogOpChanges(self, op: Op) -> None:
        store = self.getFrame(op.type)
        changedCols = op.changed.columns.get_level_values(0).unique()
        self._invalidateCached(op.changed.index, op.type, changedCols)
        store._state.increment()


SOURCE = LazyGeoPandas()


class SharedState:
    version: int

    def __init__(self) -> None:
        self.version = 0

    def increment(self):
        self.version += 1

    def __copy__(self):
        return self


T = TypeVar("T", bound=LazyGeoPandas)


class LazyGeoFrame(Generic[T]):
    _rootDf: gp.GeoDataFrame
    _state: SharedState
    _currentVersion: int
    _filterIdx: pd.Index
    _schema: Schema
    _store: T
    _columns: list[str]
    _computingColumns: list[list[str]]

    def __init__(self, schema: Schema, data: gp.GeoDataFrame = None, store: T = SOURCE) -> None:
        self._schema = schema
        if data is None:
            data = gp.GeoDataFrame()
        self._rootDf = schema.setColumnTypes(data)
        self._store = store
        self._columns = []
        self._filterIdx = None
        self._state = SharedState()
        self._currentVersion = -1
        self._computingColumns = []
        self.updateColumns()
        store.addSchema(self)

    def updateColumns(self):
        self._columns = []
        for attr in self._schema._attributes.values():
            if attr["key"] in self._schema._index:
                continue
            self._columns.append(attr["key"])

    def getStore(self) -> T:
        return self._store

    @property
    def _df(self):
        if self._filterIdx is None:
            return self._rootDf

        if self._state.version != self._currentVersion:
            self._dfc = self._rootDf.loc[self._filterIdx]
            self._currentVersion = self._state.version

        return self._dfc

    @property
    def index(self):
        return self._df.index

    def loadData(self, data: gp.GeoDataFrame) -> None:
        self._rootDf = self._schema.setColumnTypes(data)

    def addComputed(self, column: str, attribute: ColumnAttributes, func: Callable[[], Union[gp.GeoSeries, gp.GeoDataFrame]], dependencies: Union[List[str], dict[str, list[str]]] = {}, skipUpdate=False) -> None:
        attributes = _ColumnAttributes.normalize({
            "_dependencies": dependencies,
            **attribute,
            "key": column,
            "_func": func,
        }, self._schema._key)

        self._schema._addAttribute(column, attributes)
        if skipUpdate:
            return
        self.updateComputed()

    def updateComputed(self):
        self.updateColumns()
        self._store.updateDependents(self)

    def getFrame(self, key: str):
        return self._store.getFrame(key)
    
    def pendingColumns(self) -> list[str]:
        if len(self._computingColumns) == 0:
            return []
        return self._computingColumns[-1]

    @property
    def columns(self):
        return self._columns

    @property
    def columnsAttributes(self) -> Dict[str, ColumnAttributes]:
        return self._schema._attributes

    def _filter(self, index: pd.Index):
        filtered = copy(self)
        if isinstance(index, pd.Series):
            index = index[index].index
        elif not isinstance(index, slice) and not isinstance(index, list):
            index = slice(index, index)

        df = self._df.loc[index]
        if isinstance(df, pd.Series):
            df = df.to_frame().T
        filtered._filterIdx = df.index

        return filtered

    def _parseKeyRow(self, items):
        row = None
        key = None

        if isinstance(items, slice) or isinstance(items, pd.Index) or isinstance(items, pd.Series) or isinstance(items, np.ndarray):
            row = items
        elif isinstance(items, tuple):
            if len(items) == 2 and not (isinstance(items[1], int) or isinstance(items[1], np.int64) or isinstance(items[1], np.int32)):
                row = items[0]
                key = items[1]
            else:
                row = items
        elif isinstance(items, int) or isinstance(items, np.int64) or isinstance(items, np.int32):
            row = items
        elif isinstance(items, str):
            key = items
        elif isinstance(items, list) and len(items) > 0:
            if isinstance(items[0], str):
                key = items
            elif isinstance(items[0], bool):
                row = items
        else:
            raise ValueError("Invalid item type.", type(items))

        return row, key

    def __getitem__(self, items):
        row, key = self._parseKeyRow(items)

        filtered = self
        if row is not None:
            filtered = self._filter(row)
            if key is None and not (isinstance(row, slice) and row == slice(None, None, None)):
                return filtered

        if isinstance(key, str):
            filtered._insureComputed([key])
        else:
            if not isinstance(key, list):
                key = filtered.columns
            filtered._insureComputed(key)

        df: pd.DataFrame = filtered._df.loc[:, key]
        if df.shape[0] < 1:
            if (isinstance(row, tuple) and df.index.nlevels == len(row)) or df.index.nlevels == 1 and self._schema.isIndexType(row):
                if df.empty:
                    return None
                return df.values[0]
        return df

    def undo(self) -> None:
        self._store.undo()

    def redo(self) -> None:
        self._store.redo()

    def drop(self, id: Union[Hashable, Sequence[Hashable], pd.Index], skipLog=False) -> None:
        self._store._drop(self._schema._key, id, skipLog)

    def update(self, ids: Union[Hashable, Sequence[Hashable], pd.Index], value: Schema, replaceLog=False, skipLog=False) -> None:
        self._store._update(self._schema._key, ids, value, replaceLog, skipLog)

    def invalidClone(self, depKey: str) -> Union[None, Self]:
        if depKey not in self._df.columns:
            return None if self._df.empty else self
        else:
            filter = self._df.loc[:, depKey] != True
            if not filter.any():
                return None
            invalid = self._df[filter]

        if invalid.empty:
            return None

        if self._df.shape[0] == invalid.shape[0]:
            return self

        filtered = copy(self)
        filtered._filterIdx = invalid.index
        return filtered

    def _insureComputed(self, columns: Iterator[str]) -> None:
        attributes = self._schema._attributes
        df = self._store.getFrame(self._schema._key)._df
        computed = set()
        try:
            self._computingColumns.append(list(columns))
            for column in columns:
                if not column in attributes:
                    continue

                attribute = attributes[column]

                if not "_func" in attribute:
                    continue

                if column in computed:
                    continue  # Already computed

                depKey = column + ".valid"
                invalidClone = self.invalidClone(depKey)

                if invalidClone is None:
                    continue

                # Compute all the dependencies
                for depStore, deps in attribute["_dependencies"].items():
                    if depStore == self._schema._key:
                        invalidClone._insureComputed(deps)
                        continue

                    store = self._store._frames[depStore]
                    ids = invalidClone._schema._mapIds(depStore, store._df)
                    storeClone = copy(store)
                    storeClone._filterIdx = store._df.loc[ids].index
                    storeClone._insureComputed(deps)

                results = attribute["_func"](invalidClone)

                missingIndex = invalidClone._df.index
                if isinstance(results, pd.DataFrame):
                    computed.update(results.columns)
                    depKey = [c + ".valid" for c in results.columns]
                    df.loc[results.index, results.columns] = results.values
                else:
                    df.loc[missingIndex, column] = results

                self._state.increment()

                if len(attribute["_dependencies"]) != 0:
                    df.loc[missingIndex, depKey] = True
        finally:
            self._computingColumns.pop()

    def _getDependentColumns(self, columns: Iterator[str]) -> dict[str, Set[str]]:
        invalidate: dict[str, set[str]] = {}

        if not self._schema._key in self._store._dependents:
            return invalidate

        dependents = self._store._dependents[self._schema._key]

        for column in columns:
            if not column in dependents:
                continue

            depsStore = dependents[column]
            for dependencyKey, invalidColumns in depsStore.items():
                if not dependencyKey in invalidate:
                    invalidate[dependencyKey] = copy(invalidColumns)
                else:
                    invalidate[dependencyKey].update(invalidColumns)

        return invalidate

    def toBytes(self):
        return toBytes(self._rootDf)


class LazyGeoSeries(LazyGeoFrame[T]):
    def __init__(self, schema: Schema, data: gp.GeoSeries = None, store: T = SOURCE) -> None:
        if not data is None:
            data = data.to_frame(name=0).T
        super().__init__(schema, data, store)
        self._fillMissing()

    def _fillMissing(self):
        if not (0 in self._rootDf.index):
            self.drop(True)

        current = self._rootDf.loc[0, :].to_dict()
        if "modified" in current:
            del current["modified"]
        self.update(self._schema.withDefaults(**current), skipLog=True)

    def __getitem__(self, items):
        series = super().__getitem__((0, items))
        return series.loc[0]

    def drop(self, skipLog=False) -> None:
        # clear the data and use defaults
        self.update(self._schema.withDefaults(), skipLog=skipLog)

    def update(self, value: Schema, replaceLog=False, skipLog=False) -> None:
        return super().update(0, value, replaceLog, skipLog)


def mergeDeps(a: dict[str, set[str]],  b: dict[str, set[str]]) -> bool:
    changed = False
    for key, value in b.items():
        if key in a:
            preLen = len(a[key])
            a[key].update(value)
            changed = changed or preLen != len(a[key])
        else:
            a[key] = value
            changed = True
    return changed


def toBytes(df: gp.GeoDataFrame):
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)
