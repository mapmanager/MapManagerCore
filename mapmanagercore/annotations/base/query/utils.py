import numpy as np
import shapely
import pandas as pd
from typing import Callable, Dict, List
from pandas.util import hash_pandas_object
from copy import copy
from ....benchmark import timeAll, timer


class Query:
    def __init__(self, title: str, func: Callable[[], pd.Series], categorical: bool = False, idx: int = None):
        self.title = title
        self.categorical = categorical
        self.func = func
        self.idx = idx

    @timer
    def runWith(self, annotations, insureCached=False) -> pd.Series:
        result = self.func(annotations, insureCached=insureCached)

        if insureCached:
            return
        return result if self.idx is None else result[self.idx]

    def isCategorical(self) -> bool:
        return self.categorical

    def getTitle(self) -> str:
        return self.title


class QueryableInterface:
    PLOT_ABLE_QUERIES: List[Query] = []
    QUERIES: List[Query] = []
    QUERIES_MAP: Dict[str, Query] = {}

    def queries(self) -> List[Query]:
        return self.PLOT_ABLE_QUERIES

    def runQuery(self, query: Query) -> pd.Series:
        return query.runWith(self)

    @timeAll
    def table(self, queries: List[Query] = PLOT_ABLE_QUERIES) -> pd.DataFrame:
        columns = {}

        for query in queries:
            data = query.runWith(self)
            title = query.getTitle()
            if isinstance(data, pd.DataFrame):
                for column in data.columns:
                    columns[f"{title} {column}"] = data[column]
            else:
                columns[title] = data
        return pd.DataFrame(columns)

    def dataFrame(self, queries: List[Query] = QUERIES) -> pd.DataFrame:
        return self.table(queries)

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
            newModDate = np.maximum(
                self._lineSegments["modified"][self._points["segmentID"]].values, newModDate)

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
            df["segmentHash"] = segmentsHash[df["segmentID"]].values

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


def queryable(title: str = None, categorical: bool = False, dependencies: List[str] = None, segmentDependencies: List[str] = None, plotAble: bool = True):
    # A queryable function can return either a pandas Series or DataFrame.
    # If multiple values can be computed more efficiently together
    # they should be implemented in a single function and returned as a DataFrame.
    # If it returns a DataFrame, the columns will be prefixed with the query name.
    def wrapper(func):
        fullTitle = title
        if fullTitle is None:
            fullTitle = func.__name__

        key = func.__name__

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

        query = Query(fullTitle, finalFunc, categorical)
        QueryableInterface.QUERIES_MAP[key] = query

        if plotAble:
            QueryableInterface.PLOT_ABLE_QUERIES.append(query)
        QueryableInterface.QUERIES.append(query)

        return finalFunc
    return wrapper


# bug fix for shapely hash
shapely.geometry.base.BaseGeometry.__hash__ = lambda x: hash(x.wkb)
