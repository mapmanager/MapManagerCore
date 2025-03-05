from typing import Any, Callable, Dict, List, Optional, Self, TypeVar, TypedDict, Union
import pandas as pd
from ..config import Colors, Symbol  # abb Color to Colors
from typing import List, Dict, Any, Union


class ColumnAttributes(TypedDict):
    """
    Represents the attributes of a column in a GeoDataFrame.

    Attributes:
        title (str): The title of the column.
        categorical (bool): Indicates whether the column is categorical or not.
        divergent (bool): Indicates whether the column is divergent or not.
        description (str): The description of the column.
        group (str): The group to which the column belongs.
        colors (Union[List[Color], Dict[Any, Color]]): The colors associated with the column.
        symbols (Union[List[Symbol], Dict[Any, Symbol]]): The symbols associated with the column.
        plot (bool): Indicates whether the column should be plotted or not.
        version(int): The version of the computed column.
        type (str): The type of the column.
    
    Notes:
        When there is a new version of the column, the version number should be incremented.
        This will cause the column to be recomputed if it is a computed using a prior version.
    """

    title: str
    categorical: bool
    divergent: bool
    description: str
    group: str
    colors: Union[List[Colors], Dict[Any, Colors]]  # abb Color to Colors
    symbols: Union[List[Symbol], Dict[Any, Symbol]]
    plot: bool
    version: int
    type: str

    def default():
        """
        Returns a default instance of ColumnAttributes.

        Returns:
            ColumnAttributes: A default instance of ColumnAttributes.

        """
        return ColumnAttributes({
            "categorical": False,
            "divergent": False,
            "plot": True,
            "description": "",
            "version": 0,
        })


class _ColumnAttributes(ColumnAttributes):
    _func: Optional[Callable[[], pd.Series]]
    _dependencies: dict[str, list[str]]
    """The column name."""
    key: str

    def normalize(attributes: ColumnAttributes, schemaKey: str) -> Self:
        """
        Normalizes the attributes of a column, and sets default values for missing attributes.
        """
        if isinstance(attributes["_dependencies"], list):
            if len(attributes["_dependencies"]) > 0:
                attributes["_dependencies"] = {
                    schemaKey: attributes["_dependencies"]}
            else:
                attributes["_dependencies"] = {}

        return _ColumnAttributes({
            **ColumnAttributes.default(),
            **attributes,
        })
