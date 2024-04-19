from typing import Any, Dict, List, TypedDict, Union
from ...config import Color, Symbol


class ColumnAttributes(TypedDict):
    title: str
    categorical: bool
    divergent: bool
    description: str
    colors: Union[List[Color], Dict[Any, Color]]
    symbols: Union[List[Symbol], Dict[Any, Symbol]]
    key: str
    plot: bool

    def default():
        return ColumnAttributes({
            "categorical": False,
            "divergent": False,
            "plot": True,
            "description": "",
        })
