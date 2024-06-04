from dataclasses import dataclass
import pandas as pd
from typing import Any, Callable, List, Self, TypeVar, Union, Unpack
import numpy as np
import geopandas as gp
from shapely.geometry.base import BaseGeometry
from .attributes import ColumnAttributes, _ColumnAttributes


class MISSING_VALUE:
    def __repr__(self):
        return "unassigned"

    def __str__(self):
        return "unassigned"


MISSING_VALUE = MISSING_VALUE()


class Schema:
    def __init_subclass__(cls) -> None:
        cls._attributes: dict[str, _ColumnAttributes] = {}
        cls._annotations = cls.__annotations__
        cls._key = cls.__name__
        cls._index: Union[list[Any], Any] = []

        cls._defaults = {}
        cls._relationships: dict[str, list[str]] = {}
        return super().__init_subclass__()

    @classmethod
    def withDefaults(cls, **kwargs):
        for key, value in cls._defaults.items():
            if not key in kwargs:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def _addAttribute(cls, column: str, attribute: _ColumnAttributes):
        if not "key" in attribute:
            attribute["key"] = column
        if not "title" in attribute:
            attribute["title"] = column
        if not "group" in attribute:
            attribute["group"] = "Other"
        cls._attributes[column] = attribute

    # from key schema to current schema
    @classmethod
    def _reverseMapIds(cls, key: str, toDf: gp.GeoDataFrame, fromDf: gp.GeoDataFrame, ids: pd.Index = None):
        if not key in cls._relationships:
            return slice(None)

        keys = cls._relationships[key]

        if not ids is None:
            fromDf = fromDf.loc[ids, :]
        return toDf.join(fromDf, on=keys, how='inner', lsuffix='_from', rsuffix='_to').index

    # from current schema to key schema
    @classmethod
    def _mapIds(cls, key: str, df: gp.GeoDataFrame, ids: pd.Index = None):
        if not key in cls._relationships:
            return slice(None)

        keys = cls._relationships[key]

        if not ids is None:
            df = df.loc[ids, :]

        found = df.reset_index()[keys]

        if len(keys) > 1:
            return pd.MultiIndex.from_frame(found)

        return pd.Index(found.iloc[:, 0].values)

    @classmethod
    def setColumnTypes(cls, df: pd.DataFrame) -> gp.GeoDataFrame:
        defaults = cls._defaults
        types = cls._annotations
        df = gp.GeoDataFrame(df)
        for key, valueType in types.items():

            if hasattr(valueType, "__args__"):
                valueType = valueType.__args__[0]

            if issubclass(valueType, np.datetime64):
                valueType = "datetime64[ns]"

            if key in df.index.names:
                if int == valueType:
                    valueType = 'Int64'

                if len(df.index.names) == 1:
                    df.index = df.index.astype(valueType)
                else:
                    i = df.index.names.index(key)
                    df.index = df.index.set_levels(
                        df.index.levels[i].astype(valueType), level=i)
                continue
            if not isinstance(valueType, str) and issubclass(valueType, BaseGeometry):
                if key in df.columns and len(df[key]) > 0:
                    if not isinstance(df[key].iloc[0], BaseGeometry):
                        df[key] = gp.GeoSeries.from_wkt(df[key])
                else:
                    df[key] = gp.GeoSeries()
            else:
                if int == valueType:
                    valueType = 'Int64'
                    if key in df.columns:
                        df[key] = np.trunc(df[key])

                df[key] = df[key].astype(
                    valueType) if key in df.columns else pd.Series(dtype=valueType)

            if key in defaults:
                df.loc[:, key] = df.loc[:, key].fillna(defaults[key])

        if df.index.nlevels != len(cls._index):
            if len(cls._index) != 0:
                df.set_index(cls._index, inplace=True, drop=True)
                if df.index.nlevels > 1:
                    df.sort_index(level=0, inplace=True)

        return df

    @classmethod
    def isIndexType(cls, value: Any, i=0) -> bool:
        expectedType = cls._annotations[cls._index[i]]
        return isInstanceExtended(value, expectedType)

    @classmethod
    def validateColumns(cls, values: dict[str, any], dropIndex: bool = False):
        typeColumns = cls._annotations
        if dropIndex:
            for key in cls._index:
                if key in values:
                    values.pop(key)

        for key, value in values.items():
            if not key in typeColumns:
                raise ValueError(f"Invalid column {key}")
            expectedType = typeColumns[key]
            if not isInstanceExtended(value, expectedType):
                try:
                    values[key] = expectedType(value)
                    return
                except:
                    raise ValueError(f"Invalid type for column {key}")


def isInstanceExtended(value, expectedType):
    if expectedType == int and isinstance(value, np.int64):
        return True

    if hasattr(expectedType, "__args__"):
        return any(isInstanceExtended(value, ty) for ty in expectedType.__args__)

    return isinstance(value, expectedType)


def schema(index: Union[list[Any], Any], relationships: dict[Schema, dict[str, list[str]]] = {}, properties: dict[str, ColumnAttributes] = {}):
    T = TypeVar('T')

    def classWrapper(cls: T):

        defaults = {
            key: getattr(cls, key) for key in cls.__annotations__.keys() if hasattr(cls, key)
        }

        # default to None to detect missing values
        for key in cls.__annotations__.keys():
            setattr(cls, key, MISSING_VALUE)

        cls = dataclass(cls)
        cls2 = type(cls.__name__, (Schema, cls, ), {})

        cls2._annotations = cls.__annotations__
        cls2._index = index if isinstance(index, list) else [index]
        cls2._relationships = {
            key if isinstance(key, str) else key.__name__: val for key, val in relationships.items()}

        cls2._defaults = defaults
#         keys = []
#         for value, vType in cls2._annotations.items():
#             if value in defaults:
#                 keys.append(f"{value}: {vType.__name__} = d{value}")
#                 continue
#             keys.insert(0, f"{value}: {vType.__name__}")
#         funcDef = f"""
# def withDefaults(cls, {str.join(", ", keys)}):
#     return cls({str.join(", ", [f"{value}={value}"for value in cls2._annotations.keys()])})
#         """
#         globalsV = {vType.__name__: vType for value,
#                     vType in cls2._annotations.items()}
#         for key, value in defaults.items():
#             globalsV[f"d{key}"] = value
#         localsV = {}
#         exec(funcDef, globalsV, localsV)
#         cls2.withDefaults = classmethod(localsV["withDefaults"])

        for key, val in properties.items():
            cls2._addAttribute(key, _ColumnAttributes.normalize({
                **val,
                "key": key,
                "_dependencies": {},
            }, cls2.__name__))

        for key in cls2._annotations.keys():
            if not key in cls2._attributes:
                cls2._addAttribute(key, _ColumnAttributes.normalize({
                    "title": key,
                    "key": key,
                    "_dependencies": {},
                }, cls2.__name__))

        for name, method in cls.__dict__.items():
            if not hasattr(method, "_attributes"):
                continue

            cls2._addAttribute(name, _ColumnAttributes.normalize(
                method._attributes, cls2.__name__))

        return cls2
    return classWrapper


def seriesSchema(relationships: dict[Schema, dict[str, list[str]]] = {}, properties: dict[str, ColumnAttributes] = {}):
    return schema([], relationships, properties)


def calculated(dependencies: Union[List[str], dict[str, list[str]]] = {}, **attributes: Unpack[ColumnAttributes]):
    def wrapper(func: Callable[[], Union[pd.Series, pd.DataFrame]]):
        func._attributes = {
            "key": func.__name__,
            **attributes,
            "_func": func,
            "_dependencies": dependencies,
        }

        return func
    return wrapper
