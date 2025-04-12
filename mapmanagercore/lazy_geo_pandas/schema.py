from copy import copy
import types
import pandas as pd
from typing import Any, Callable, List, Self, TypeVar, Union, Unpack
import numpy as np
import geopandas as gp
from shapely.geometry.base import BaseGeometry
from .attributes import ColumnAttributes, _ColumnAttributes

from mapmanagercore.logger import logger


class MISSING_VALUE_CLASS:
    """
    Represents a missing/unset value.
    """
    _default = None

    def __init__(self, default=None):
        self._default = default

    def __repr__(self):
        return "unassigned"

    def __str__(self):
        return "unassigned"


MISSING_VALUE = MISSING_VALUE_CLASS()


class Schema:
    def __init_subclass__(cls):
        cls._attributes: dict[str, _ColumnAttributes] = {}
        cls._annotations = cls.__annotations__
        cls._key = cls.__name__
        cls._index: Union[list[Any], Any] = []

        cls._relationships: dict[str, list[str]] = {}
        return super().__init_subclass__()

    def __init__(self):
        pass

    def defaults(self) -> Self:
        """
        Returns a copy of the schema with default values set for missing values.
        """
        clone = copy(self)
        for key in clone.__annotations__.keys():
            item = getattr(clone, key)
            if isinstance(item, MISSING_VALUE_CLASS):
                setattr(clone, key, item._default)
        return clone

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
        defaults = cls().defaults()
        types = cls._annotations
        df = gp.GeoDataFrame(df)
        for key, valueType in types.items():
            if valueType.__name__ != "Tuple":
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
                        valueType) if key in df.columns and not df.empty else pd.Series(dtype=valueType)
            else:
                df[key] = df[key].astype(
                    "object") if key in df.columns and not df.empty else pd.Series(dtype="object")

            if hasattr(defaults, key):
                default = getattr(defaults, key)
                if not isinstance(default, MISSING_VALUE_CLASS):
                    if isinstance(default, tuple):
                        if key in df.columns:
                            # abb 20250411 error on load
                            try:
                                df.loc[:, key] = df.loc[:, key].apply(
                                    lambda x: x if not pd.isna(x) else default)
                            except (ValueError) as e:
                                logger.error('abb 202504')
                                logger.error(e)
                                logger.error(f'  key:{key} {type(key)}')
                                logger.error(f'  default:{default} {type(default)}')

                        else:
                            df.loc[:, key] = df.apply(
                                lambda x: default, axis=1)
                    else:
                        df.loc[:, key] = df.loc[:, key].fillna(default)

        if df.index.nlevels != len(cls._index):
            if len(cls._index) != 0:
                # logger.info('drop=True')
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
        if not cls._index:
            # no index was set (Series schema)
            return False

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
                    # abb this is throwing
                    # TypeError: int() argument must be a string, a bytes-like object or a real number, not 'dict'
                    values[key] = expectedType(value)
                    return
                except Exception as e:
                    print(e)
                    raise ValueError(f"Invalid type for column {key}")


def isInstanceExtended(value, expectedType):
    """
    Checks if the value is of the expected type.
    Also checks for numpy int64 type.
    """
    if expectedType == int and isinstance(value, np.int64):
        return True

    if expectedType.__name__ == "Tuple" and isinstance(value, tuple):
        return True

    if hasattr(expectedType, "__args__"):
        return any(isInstanceExtended(value, ty) for ty in expectedType.__args__)

    return isinstance(value, expectedType)


def schema(index: Union[list[Any], Any] = [], relationships: dict[Schema, dict[str, list[str]]] = {}):
    """
    A decorator to define a schema class.

    Args:
        index (Union[list[Any], Any]): The index of the schema rows. Multi index can be used by passing a list of column names. If no index is provided the schema will be treated as a series schema.
        relationships (dict[Schema, dict[str, list[str]]]): The relationships between this schema and other schemas. This allows the schema to track changes across schemas for computed columns with dependencies.
    """
    # TODO: Automatically infer relationships from the name and indexes of schemas
    # For example if schema A  has an index of ['t', 'segmentId'] and schema B has an index of ['t', 'spineId'] and a column "segmentId". Then we can infer the relationship automatically {"Segment": ["segmentID", "t"]}
    # We can also infer dependencies by dry running computed columns post all schema initializations
    # In essence we can pass in empty subclasses of a frame that collects all the columns that are accesses along with the cross dependencies in the schema.
    # To detect dependencies across schemas we can override all active frames temporarily to collect columns across different schemas.
    # Note this must be done on boot up caution must be taken to avoid multi threading issues.
    T = TypeVar('T', bound=Schema)

    def classWrapper(cls: T) -> T:
        field_attributes = {}

        # Extract field_attributes from fields
        for key, fieldType in cls.__annotations__.items():
            if hasattr(cls, key):
                field = getattr(cls, key)
                if isinstance(field, Field):
                    if not "type" in field.attributes:
                        fieldType = fieldType if fieldType else field._default.__class__
                        field.attributes["type"] = fieldType.__name__
                    field_attributes[key] = field.attributes

        cls2: Schema = cls
        cls2._relationships = {
            key if isinstance(key, str) else key.__name__: val for key, val in relationships.items()}
        cls2._annotations = cls.__annotations__
        cls2._index = index if isinstance(index, list) else [index]

        for key, val in field_attributes.items():
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


class Field(MISSING_VALUE_CLASS):
    def __init__(self, default: Any, attributes: Unpack[ColumnAttributes]):
        super().__init__(default)
        self.attributes = attributes

    __class_getitem__ = classmethod(types.GenericAlias)


U = TypeVar('U')


def field(default: U = MISSING_VALUE, **attributes: Unpack[ColumnAttributes]) -> U:
    """
    A decorator to define a field in a schema class.

    Args:
        default (Any): The default value of the field.
        title (str): The title of the column.
        categorical (bool): Indicates whether the column is categorical or not.
        divergent (bool): Indicates whether the column is divergent or not.
        description (str): The description of the column.
        group (str): The group to which the column belongs.
        colors (Union[List[Color], Dict[Any, Color]]): The colors associated with the column.
        symbols (Union[List[Symbol], Dict[Any, Symbol]]): The symbols associated with the column.
        plot (bool): Indicates whether the column should be plotted or not.
        type (str): The type of the column.
    """
    return Field(default, attributes)


def compute(dependencies: Union[List[str], dict[str, list[str]]] = {}, **attributes: Unpack[ColumnAttributes]):
    """
    A decorator to define a method that computes a column in the schema.

    Args:
        dependencies (Union[List[str], dict[str, list[str]]]): 
            The dependencies of the computed column. 
            Use a dictionary to specify dependencies across multiple schemas with the schema name being the key and an array of dependency columns being the column.
            An array to specify dependencies within the same schema.
            The dependencies for the computed column. Defaults to {}.
            The dictionary can be used to specify dependencies across different schemas.
            {"schemaName": ["column1", "column2"], "schemaName2": ["column3", "column4"]}
        title (str): The title of the column.
        categorical (bool): Indicates whether the column is categorical or not.
        divergent (bool): Indicates whether the column is divergent or not.
        description (str): The description of the column.
        group (str): The group to which the column belongs.
        colors (Union[List[Color], Dict[Any, Color]]): The colors associated with the column.
        symbols (Union[List[Symbol], Dict[Any, Symbol]]): The symbols associated with the column.
        plot (bool): Indicates whether the column should be plotted or not.
        version(int): The version of the computed column. When a computed column is updated, the version should be incremented so that the older versions of the columns are automatically invalidated and are recomputed.
        type (str): The type of the column.
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
