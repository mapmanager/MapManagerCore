from typing import Union
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely import force_2d
import skimage.draw


def filterMask(d, index_filter):
    if index_filter == None or len(index_filter) == 0:
        return np.full(len(d), False)
    return ~d.isin(index_filter)


def shapeIndexes(d: Union[Polygon, LineString]):
    d = force_2d(d)
    if isinstance(d, Polygon):
        x, y = zip(*d.exterior.coords)
        return skimage.draw.polygon(x, y)

    x, y = zip(*d.coords)
    return skimage.draw.line(int(x[0]), int(y[0]), int(x[1]), int(y[1]))


def validateColumns(values: dict[str, any], typeColumns: dict[str, type]):
    for key, value in values.items():
        if not key in typeColumns:
            raise ValueError(f"Invalid column {key}")
        expectedType = typeColumns[key]
        if isinstance(expectedType, str):
            if "datetime64[ns]" == expectedType:
                if not isinstance(value, np.datetime64):
                    raise ValueError(
                        f"Invalid type for column {key} expected {expectedType}")
        if not isinstance(value, expectedType):
            try:
                values[key] = expectedType(value)
                return
            except:
                raise ValueError(f"Invalid type for column {key}")
