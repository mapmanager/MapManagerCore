from dataclasses import dataclass
import pandas as pd
from typing import Any, Callable, List, Self, TypeVar, Union, Unpack
import numpy as np
import geopandas as gp
from shapely.geometry.base import BaseGeometry
from .attributes import ColumnAttributes, _ColumnAttributes

from mapmanagercore.logger import logger

class MISSING_VALUE:
    """
    Represents a missing/unset value.
    """

    def __repr__(self):
        return "unassigned"

    def __str__(self):
        return "unassigned"


MISSING_VALUE = MISSING_VALUE()


class Schema:
    def __init_subclass__(cls):
        cls._attributes: dict[str, _ColumnAttributes] = {}
        cls._annotations = cls.__annotations__
        cls._key = cls.__name__
        cls._index: Union[list[Any], Any] = []

        cls._defaults = {}
        cls._relationships: dict[str, list[str]] = {}
        return super().__init_subclass__()

    @classmethod
    def withDefaults(cls, **kwargs):
        """
        Creates an instance of the schema using default values when values are unset.
        """
        for key, value in cls._defaults.items():
            if not key in kwargs:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def _addAttribute(cls, column: str, attribute: _ColumnAttributes):
        """
        Adds a column's attributes to the schema
        """
        if not "key" in attribute:
            attribute["key"] = column
        if not "title" in attribute:
            attribute["title"] = column
        if not "group" in attribute:
            attribute["group"] = "Other"
        cls._attributes[column] = attribute

    @classmethod
    def _reverseMapIds(cls, key: str, toDf: gp.GeoDataFrame, fromDf: gp.GeoDataFrame, ids: pd.Index = None):
        """
        Maps the ids from the key schema to the current schema.
        Used to track changes across schemas.
        If no relationships are defined between the schemas, it will return all ids.
        """
        if not key in cls._relationships:
            return slice(None)

        keys = cls._relationships[key]

        if not ids is None:
            fromDf = fromDf.loc[ids, :]
        return toDf.join(fromDf, on=keys, how='inner', lsuffix='_from', rsuffix='_to').index

    @classmethod
    def _mapIds(cls, key: str, df: gp.GeoDataFrame, ids: pd.Index = None):
        """
        Maps the ids from the current schema to the key schema.
        Used to track changes across schemas.
        If no relationship are defined between the schemas, it will return all ids.
        """
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
        """
        Sets the column types of the dataframe to the types defined by the schema class.
        """
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
                logger.info('drop=True')
                df.set_index(cls._index, inplace=True, drop=True)
                if df.index.nlevels > 1:
                    df.sort_index(level=0, inplace=True)

        return df

    @classmethod
    def isIndexType(cls, value: Any, level=0) -> bool:
        """
        Checks if the value is of the type defined in the schema's index.

        Args:
            value (Any): The value to be checked.
            level (int): The index level to be checked.

        Returns:
            bool: True if the value is of the type defined in the schema's index, False otherwise.
        """
        expectedType = cls._annotations[cls._index[level]]
        return isInstanceExtended(value, expectedType)

    @classmethod
    def validateColumns(cls, values: dict[str, any], dropIndex: bool = False):
        """
        Validates the values to insure they are consistent with the schema.

        Args:
            values (dict[str, any]): The values to be validated.
            dropIndex (bool): If True, the index columns will be removed from the values.
        """

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
    """
    Checks if the value is of the expected type.
    Also checks for numpy int64 type.
    """
    if expectedType == int and isinstance(value, np.int64):
        return True

    if hasattr(expectedType, "__args__"):
        return any(isInstanceExtended(value, ty) for ty in expectedType.__args__)

    return isinstance(value, expectedType)


def schema(index: Union[list[Any], Any], relationships: dict[Schema, dict[str, list[str]]] = {}, properties: dict[str, ColumnAttributes] = {}):
    """
    A decorator to define a schema class.

    Args:
        index (Union[list[Any], Any]): The index of the schema.
        relationships (dict[Schema, dict[str, list[str]]]): The relationships between this schema and other schemas.
        properties (dict[str, ColumnAttributes]): The properties of the fields defined by the schema.
    """
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
    """
    A decorator to define a schema class for a series.
    Differs from schema in that it defines a schema for a series instead of a frame.
    
    Args:
        relationships (dict[Schema, dict[str, list[str]]]): The relationships between this schema and other schemas.
        properties (dict[str, ColumnAttributes]): The properties of the fields defined by the schema.
    """
    return schema([], relationships, properties)


def compute(dependencies: Union[List[str], dict[str, list[str]]] = {}, **attributes: Unpack[ColumnAttributes]):
    """
    A decorator to define a method that computes a column in the schema.
    
    Args:
        dependencies (Union[List[str], dict[str, list[str]]]): The dependencies of the method.
        attributes (Unpack[ColumnAttributes]): The attributes of the computed column.
    """
    def wrapper(func: Callable[[], Union[pd.Series, pd.DataFrame]]):
        func._attributes = {
            "key": func.__name__,
            **attributes,
            "_func": func,
            "_dependencies": dependencies,
        }

        return func
    return wrapper
