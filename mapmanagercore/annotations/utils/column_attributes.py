from typing import Any, Dict, List, TypedDict, Union


class ColumnAttributes(TypedDict):
    title: str
    categorical: bool
    description: str
    colors: Union[List[str], Dict[Any, str]]
    key: str
    plot: bool

    def default():
        return ColumnAttributes({
            "categorical": False,
            "plot": True,
            "description": "",
        })

