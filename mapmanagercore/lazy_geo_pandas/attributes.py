from typing import Any, Callable, Dict, List, Optional, Self, TypedDict, Union
import pandas as pd

from ..config import Color, Symbol


class ColumnAttributes(TypedDict):
    title: str
    categorical: bool
    divergent: bool
    description: str
    colors: Union[List[Color], Dict[Any, Color]]
    symbols: Union[List[Symbol], Dict[Any, Symbol]]
    plot: bool

    def default():
        return ColumnAttributes({
            "categorical": False,
            "divergent": False,
            "plot": True,
            "description": "",
        })


class _ColumnAttributes(ColumnAttributes):
    _func: Optional[Callable[[], pd.Series]]
    _dependencies: dict[str, list[str]]
    key: str

    def normalize(attributes: ColumnAttributes, schemaKey: str) -> Self:
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
