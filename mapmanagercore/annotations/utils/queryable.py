import numpy as np
import shapely
import pandas as pd
import geopandas as gp
from typing import Callable, Dict, List, Unpack
from pandas.util import hash_pandas_object
from copy import copy
from .column_attributes import ColumnAttributes
from ...benchmark import timeAll, timer


class Query:
    def __init__(self, key: str, func: Callable[[], pd.Series], aggregate: List[str] = None, **kwargs):
        self.func = func
        self.key = key
        self.attr = ColumnAttributes(
            {**ColumnAttributes.default(), **kwargs, "column": key})
        self.aggregate = aggregate

    @timer
    def runWith(self, annotations, insureCached=False) -> pd.Series:
        result = self.func(annotations, insureCached=insureCached)
        if insureCached:
            return
        return result


class QueryableInterface:
    QUERIES_MAP: Dict[str, Query] = {}

    @property
    def index(self):
        return self._points.index

    @property
    def columns(self) -> List[str]:
        columns = []
        for query in self.QUERIES_MAP.values():
            if query.aggregate is None:
                columns.append(query.key)
            else:
                for agg in query.aggregate:
                    for channel in range(0, self.images.channels()):
                        columns.append(f"{query.key}_ch{channel}_{agg}")

        return columns

    def _ipython_key_completions_(self):
        return self.columns

    @property
    def columnsAttributes(self) -> Dict[str, ColumnAttributes]:
        attributes = {}
        for query in self.QUERIES_MAP.values():
            if query.aggregate is None:
                attributes[query.key] = query.attr
            else:
                for agg in query.aggregate:
                    for channel in range(0, self.images.channels()):
                        queryKey = f"{query.key}_ch{channel}_{agg}"
                        attributes[queryKey] = {
                            **query.attr,
                            "key": queryKey,
                            "title": f"{query.attr['title']} Channel {channel} ({agg})"
                        }

        return attributes

    def _table(self, columns: List[str]) -> pd.DataFrame:
        result = {}

        for column in columns:
            aggregate = not column in self.QUERIES_MAP
            if aggregate:
                query = self.QUERIES_MAP[column.split("_")[0]]
            else:
                query = self.QUERIES_MAP[column]

            data = query.runWith(self)
            if isinstance(data, pd.DataFrame):
                if aggregate:
                    result[column] = data[column]
                    continue

                for column in data.columns:
                    result[column] = data[column]
            else:
                result[query.key] = data

        for column in columns:
            if column not in result or result[column].empty:
                continue
            if isinstance(result[column].iloc[0], shapely.geometry.base.BaseGeometry):
                result[column] = gp.GeoSeries(result[column])

        return gp.GeoDataFrame(result)

    def _filter(self, index: pd.Index):
        filtered = copy(self)

        if isinstance(index, str):
            index = [index]

        if isinstance(index, tuple) and len(index) == 2:
            index = [index]

        filtered._points = filtered._points.loc[index, :]
        return filtered

    @timeAll
    def __getitem__(self, items):
        row = None
        key = None
        filtered = self
        if isinstance(items, slice) or isinstance(items, pd.Index) or isinstance(items, pd.Series) or isinstance(items, np.ndarray):
            row = items
        elif isinstance(items, tuple):
            if len(items) > 1:
                row = items[0]
                key = items[1]
            else:
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

        if row is not None:
            filtered = self._filter(row)
            if key is None and not isinstance(row, slice):
                return filtered

        if isinstance(key, str):
            df = filtered._table([key])
            if df.shape[1] == 1:
                return df.iloc[:, 0]
            return df

        if isinstance(key, list):
            return filtered._table(key)

        return filtered._table(list(self.QUERIES_MAP.keys()))

    @timer
    def _updateMissingValues(self, key, results, missingIndex, depKey, newHashes):
        if isinstance(results, pd.DataFrame):
            self._points.loc[missingIndex, results.columns] = results.values
        else:
            self._points.loc[missingIndex, key] = results
        self._points.loc[missingIndex, depKey] = newHashes

    @timer
    def _getResults(self, key):
        if key in self._points:
            return self._points[key]

        columns = self._points.columns.str.startswith(key + " ")
        df = self._points[self._points.columns[columns]]
        # remove the prefix
        return df.rename(columns=lambda x: x[len(key) + 1:])

    @timer
    def _invalidEntriesModDate(self, modKey: str, segmentDependencies: List[str] = None):
        # opt avoid the cost of a hash by checking the modified data
        newModDate = self._points["modified"]
        if segmentDependencies is not None:
            index = pd.MultiIndex.from_arrays(
                [self._points['segmentID'], self._points.index.get_level_values(1)])
            newModDate = np.maximum(
                self._lineSegments["modified"][index].values, newModDate)

        invalid = None
        if modKey in self._points:

            invalid = self._points[modKey] != newModDate
            if invalid.any() == 0:
                return [None, None]

        return [newModDate, invalid]

    @timer
    def _withInvalidEntriesHash(self, depKey: str, invalid, dependencies: List[str], segmentDependencies: List[str] = None):
        # check for changes using the hash of the dependencies

        df = self._points.loc[:, self._points.columns.intersection(
            dependencies).union(["segmentID"])]

        if invalid is not None:
            df = df[invalid]

        if segmentDependencies is not None:
            segmentsHash = hash_pandas_object(
                self._lineSegments[self._lineSegments.columns.intersection(segmentDependencies)], index=False)
            index = pd.MultiIndex.from_arrays(
                [df["segmentID"], df.index.get_level_values(1)])
            df["segmentHash"] = segmentsHash[index].values

        hash = hash_pandas_object(df, index=False)
        if depKey in self._points:
            invalid = self._points.loc[hash.index, depKey] != hash
            hash = hash[invalid]

        if hash.shape[0] == 0:
            return [None, hash]

        # create a copy of the annotations that needs to be updated
        selfCopy = copy(self)
        selfCopy._points = selfCopy._points.loc[hash.index]
        return [selfCopy, hash]

    def _updateModDate(self, modKey, missingIndex, newModDate):
        self._points.loc[missingIndex, modKey] = newModDate

    def _channels(self):
        return self.images.channels()


def queryable(dependencies: List[str] = None, segmentDependencies: List[str] = None, aggregate: List[str] = None, **kwargs: Unpack[ColumnAttributes]):
    def wrapper(func):
        fullTitle = kwargs["title"] if "title" in kwargs else None
        if fullTitle is None:
            fullTitle = func.__name__

        key = func.__name__
        hasChannels = "channel" in func.__code__.co_varnames
        func = timer(func)

        if dependencies is not None or segmentDependencies is not None:
            deps = dependencies or []
            depKey = key + ".deps"
            modKey = key + ".m"

            def callCached(self: QueryableInterface, insureCached=False):
                # Look for changes using the mod date (fast)
                [newModDate, invalid] = self._invalidEntriesModDate(
                    modKey, segmentDependencies)

                # Fall back to comparing hashes (slow)
                if newModDate is not None:
                    # insure dependencies are computed
                    for dep in deps:
                        if dep in QueryableInterface.QUERIES_MAP:
                            QueryableInterface.QUERIES_MAP[dep].func(
                                self, insureCached=True)

                    [missing, newHashes] = self._withInvalidEntriesHash(
                        depKey, invalid, deps, segmentDependencies)

                    missingIndex = newHashes.index

                    if missing is not None:
                        if hasChannels:
                            channels = missing._channels()

                            def aggregateFunc(missing, channel, aggregate):
                                images = func(missing, channel=channel)
                                images = images.explode().astype(np.uint64)
                                images = images.groupby(level=0)
                                return images.aggregate(aggregate)

                            results = pd.concat([
                                aggregateFunc(missing, channel, aggregate).add_prefix(f"{key}_ch{channel + 1}_") for channel in range(0, channels)
                            ], axis=1)
                        else:
                            # compute missing values
                            results = func(missing)

                        if isinstance(results, pd.DataFrame):
                            results = results.add_prefix(key + " ")

                        self._updateMissingValues(
                            key, results, missingIndex, depKey, newHashes)

                    if newModDate is not None:
                        self._updateModDate(modKey, missingIndex, newModDate)

                if insureCached:
                    return

                return self._getResults(key)

            finalFunc = callCached
        else:
            def finalFunc(self: QueryableInterface, insureCached=False):
                return func(self)

        query = Query(key, finalFunc, aggregate, **kwargs)
        QueryableInterface.QUERIES_MAP[key] = query

        return finalFunc
    return wrapper


# bug fix for shapely hash
shapely.geometry.base.BaseGeometry.__hash__ = lambda x: hash(x.wkb)
