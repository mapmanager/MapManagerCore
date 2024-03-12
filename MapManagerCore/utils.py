from typing import List
import numpy as np
import pandas as pd
from shapely import wkt
import geopandas as gp
from shapely.geometry import Polygon
import skimage.draw

def toGeoData(data: pd.DataFrame, geometryCols: List[str]):
    """
    Reads a CSV file with geometry columns from the given path into a geopandas GeoDataFrame.

    Args:
        path (str): The path to the CSV file.
        geometryCols (list): The list of column names containing geometry data.

    Returns:
        gp.GeoDataFrame: The loaded CSV data as a geopandas GeoDataFrame.
    """
    
    for column in geometryCols:
        data[column] = data[column].apply(wkt.loads)
    df = gp.GeoDataFrame(data, geometry=geometryCols[0])
    
    for column in geometryCols:
        df[column] = gp.GeoSeries(df[column])
    return df


def filterMask(d, index_filter):
    if index_filter == None or len(index_filter) == 0:
        return np.full(len(d), False)
    return ~d.isin(index_filter)


def polygonIndexes(d: Polygon):
    x, y = zip(*d.exterior.coords)
    return skimage.draw.polygon(x, y)
