"""
lazy_geo_pd_images/store.py
class LazyImagesGeoPandas -> _genWrappedFunc()

This _genWrappedFunc() is called by LazyImagesGeoPandas.addSchema()
"""

import numpy as np
import pandas as pd
from typing import List
import geopandas as gpd
from shapely.geometry import Polygon
from skimage.draw import polygon as skpolygon

import matplotlib.pyplot as plt

from mapmanagercore.lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader
from mapmanagercore import MapAnnotations

from mapmanagercore.logger import logger


def makeMap():
    """Make single timepoint with a synthetic image."""
    
    logger.info("1. Creating loader and importing first timepoint from ndarray ...")
    loader = mmMapLoader()
    
    # make a nparray checkerboard
    # timepoint_path = 'data/sine_2d.tif'
    imgData = _makeCheckerboard()
    
    logger.info(f'  imgData is shape: {imgData.shape}')
    timepoint_key = loader.importTimepoint(path=None,
                                           imgData=imgData)
    logger.info(f"   Timepoint key: {timepoint_key}")
    # TODO: check metadata
    
    # 2. Create MapAnnotations object
    logger.info("2. Creating MapAnnotations...")
    map_annotations = MapAnnotations(
        loader,
        lineSegments=pd.DataFrame(),
        points=pd.DataFrame()
    )

    return map_annotations

# Assuming your GeoDataFrame is called 'gdf_form1'
def convert_form1_to_form2(gdf_form1, t=1, z=0):
    """
    Convert GeoDataFrame from Form 1 to Form 2
    
    Parameters:
    -----------
    gdf_form1 : GeoDataFrame
        Input GeoDataFrame with spineID index and geometry column
    t : int, default=1
        Time point value to add
    z : int, default=0
        Z-coordinate value to add
    
    Returns:
    --------
    GeoDataFrame with multi-level index (spineID, t) and columns (shape, z)
    """
    # Reset index to make spineID a column
    gdf_temp = gdf_form1.reset_index()
    
    # Add the 't' and 'z' columns
    gdf_temp['t'] = t
    gdf_temp['z'] = z
    
    # Rename 'geometry' to 'shape'
    gdf_temp = gdf_temp.rename(columns={'geometry': 'shape'})
    
    # Set multi-level index
    gdf_form2 = gdf_temp.set_index(['spineID', 't'])
    
    return gdf_form2

# Usage example:
# gdf_form2 = convert_form1_to_form2(gdf_form1, t=1, z=0)

def addSpines(mmMap : MapAnnotations, segmentID: int):
    if segmentID == 1:
        spinePoints = [
            [150, 150],
            # [200, 600],
            # [300, 200],
            # [400, 600],
            # [500, 200],
        ]

    z = 0  # 2d images require z=0 (one image slice)

    timepointKeys = mmMap.loader.metadata.timepointKeys
    firstTimepoint = timepointKeys[0]
    tp = mmMap.getTimePoint(time=firstTimepoint)

    for idx, row in enumerate(spinePoints):
        x = row[0]
        y = row[1]
        logger.info(f'   {idx} addSpine segment:{segmentID} x:{x} y:{y}')
        
        # before we addSpine, tp.columns is empty
        logger.info('before addSpines tp.pointscolumns is:')
        print(tp.points.columns)
        logger.info('but tp.points._root is')
        print(tp.points._root)
        
        # like [segmentID, point, anchor, xBackgroundOffset, yBackgroundOffset, z, anchorZ, modified, roiExtend, roiRadius, note, userType, accept]
        logger.info('and mmMap.points._rootDf is')
        print(mmMap.points._rootDf)

        newSpineID = tp.addSpine(segmentID, x, y, z)

        # abb 20250825, omg here we go again
        # simple columns like (spineLength, spineAngle, spineSide) are NOT auto updated
        # until we access them as a key in tp.points !!!!!!!!!!!!!!!
        # check we have spineLength
        # calling tp.points['spineLength'] triggers a refresh of spineLength and nothing else???
        # logger.info(f'   1) tp EMPTY after add newSpineID {newSpineID} spineLength is:')
        # _spineLength = tp.points['spineLength']
        # print(_spineLength)

        # after addSpine() I though the underlying dataframe would be updated
        # print('tp.points._rootDf is:')
        # print(tp.points._root)
        # print('mmMap.points._rootDf is:')
        # print(mmMap.points._rootDf)
        # print('mmMap.points._rootDf.columns is:')
        # print(mmMap.points._rootDf.columns)

        # recreate tp and check again
        # tp = mmMap.getTimePoint(time=firstTimepoint)
        # logger.info(f'   2) tp EMPTY after add newSpineID {newSpineID} spineLength is:')
        # _spineLength = tp.points['spineLength']  # again this triggers refresh but other spine properties like spineAngle are still not refreshed ???
        # print(_spineLength)



def addSegment(mmMap : MapAnnotations, localSegmentID: int):
    timepointKeys = mmMap.loader.metadata.timepointKeys
    firstTimepoint = timepointKeys[0]

    tp = mmMap.getTimePoint(time=firstTimepoint)
    
    if localSegmentID == 1:
        linePoints = [
            [100, 200],
            [150, 200],
            [200, 200],
            [250, 200],
            [300, 200],
            [350, 200],
            [400, 200],
        ]

    z = 0  # 2d images require z=0 (one image slice)

    newSegmentID = tp.newSegment()
    for idx, row in enumerate(linePoints):
        x = row[0]
        y = row[1]
        # logger.info(f'   {idx} appendSegmentPoint segment:{newSegmentID} x:{x} y:{y}')
        tp.appendSegmentPoint(newSegmentID, x, y, z)

    return newSegmentID   
 
def test_intensity():
    mmMap = makeMap()
    print(mmMap)

    _localSegmentID = 1
    newSegmentID = addSegment(mmMap, _localSegmentID)
    addSpines(mmMap, newSegmentID)

    return

    #
    # compare intensity values from core vs. from gdf
    #

    firstTimepoint = 1
    
    tp = mmMap.getTimePoint(time=firstTimepoint)
    
    # for _col in tp.points.columns:
    #     print(_col)

    logger.info('fetching spineRoi_ch1 stats from backend ->')
    logger.info('  is COMBINED SPINE AND SEGMENT (e.g. points["roi"])')
    spineRoi_ch1_sum = tp.points[['spineRoi_ch1_size', 'spineRoi_ch1_sum', 'spineRoi_ch1_mean', 'spineRoi_ch1_std', 'spineRoi_ch1_min', 'spineRoi_ch1_max']]  # triggers SettingWithCopyWarning
    print(spineRoi_ch1_sum)

    # these are roi polygons, we need a computed column for roiHead_ch1_sum, ...
    # spineRoiColumns = ['roiHead', 'roiBase', 'roi']
    # dfSpineRoi = tp.points[spineRoiColumns]
    # logger.info('dfSpineRoi is:')
    # print(dfSpineRoi)

    #
    # test my simple metrics
    #
    
    # get imgData from loader
    imgData = mmMap.loader.fetchSlices(t=firstTimepoint,
                                       channelIdx=1,
                                       zRange=[0, 0])

    _roiHead = tp.points['roiHead']  # GeoSeries
    # logger.info(f'_roiHead is type: {type(_roiHead)} len: {len(_roiHead)}')
    # print(_roiHead)  # roiHead[1] corresponds to spineID label 1 (not 0)

    _roiBase = tp.points['roiBase']  # GeoSeries
    # logger.info(f'_roiBase is type: {type(_roiBase)} len: {len(_roiBase)}')
    # print(_roiBase)

    _roi = tp.points['roi']  # GeoSeries, combines roiHead and roiBase?
    # logger.info(f'_roi is type: {type(_roi)} len: {len(_roi)}')
    # print(_roi)
    """
    spineID
    1    POLYGON ((155 196, 152 196, 152 204, 168 204, ...
    Name: roi, dtype: geometry
    """
    _listOfGeoSeries = [_roiHead, _roiBase, _roi]
    # _listOfGeoSeries = [_roi]
    stats_list = []
    for item in _listOfGeoSeries:
        # item is GeoSeries
        oneStatList = [polygon_stats(poly, imgData) for poly in item]
        # print(f'oneStatList is type: {type(oneStatList)} len: {len(oneStatList)}')
        # print(oneStatList)
        stats_list.extend(oneStatList)
    
    stats_df = gpd.pd.DataFrame(stats_list)
    logger.info(f'new calculated in test stats_df:')
    print(stats_df)

    plot_img_rois(imgData, _listOfGeoSeries, stats_df['mask'])

def plot_img_rois(imgData:np.ndarray, gdf:List[gpd.GeoDataFrame], masks:list[np.ndarray]):
    """Plot an img and overlay polygons.

    Parameters:
        imgData: np.ndarray
        gdf: list of gpd.GeoDataFrame
        masks: list of np.ndarray
    Returns:
        None
    """

    if not isinstance(gdf, list):
        gdf = [gdf]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(imgData, cmap="gray", alpha=0.5)

    # Overlay polygons
    for item in gdf:
        for i, poly in enumerate(item.geometry):
            x, y = poly.exterior.xy
            ax.plot(x, y, color="red", linewidth=1)
            # label polygon with mean
            # ax.text(np.mean(x), np.mean(y), f"{gdf.loc[i,'mean']:.1f}",
            #         color="yellow", ha="center", va="center", fontsize=10,
            #         bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"))

    # Overlay masks
    maskColors = ['Reds', 'Greens', 'Blues']
    for _idx, mask in enumerate(masks):
        ax.imshow(mask, cmap=maskColors[_idx], alpha=0.5)

    # plt.title("Per-polygon mean pixel value")
    plt.show()

# Function: count pixels + extract values + compute stats
def polygon_stats(poly:Polygon, arr: np.ndarray) -> dict:

    x, y = poly.exterior.xy
    rr, cc = skpolygon(y, x, shape=arr.shape)  # y=row, x=col
    mask = np.zeros(arr.shape, dtype=bool)
    mask[rr, cc] = True
    values = arr[mask]

    if len(values) == 0:
        return {
            "count": 0,
            "sum": 0,
            "mean": np.nan,
            # "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
            'mask': mask
        }
    
    if int(np.sum(values)) == 0:
           logger.error(f'sum is 0')

    return {
        "count": len(values),
        "sum": int(np.sum(values)),
        "mean": float(np.mean(values)),
        # "median": float(np.median(values)),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "std": float(np.std(values)),
        'mask': mask
    }

def _makeCheckerboard() -> np.ndarray:
    # Example: 1000x1000 checkerboard image with 100x100 pixel squares
    _width_height = 250
    arr = np.zeros((1, _width_height, _width_height), dtype=np.uint8)
    square_size = 100
    
    for i in range(0, _width_height, square_size):
        for j in range(0, _width_height, square_size):
            # Alternate between 0 and 1 based on position
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                arr[0, i:i+square_size, j:j+square_size] = 0
            else:
                arr[0, i:i+square_size, j:j+square_size] = 3

    return arr

def fakeData():
    arr = _makeCheckerboard()

    # Example polygons in pixel coords
    # 2x polygon [spine, segment]
    polys = [    
        # spine roi
        Polygon([(146, 196), (142, 196), (142, 204), (158, 204), (158, 196), (154, 196), (154, 146), (146, 146), (146, 196)]),
        # segment roi
        # Polygon([(142, 196), (142, 204), (158, 204), (158, 196), (142, 196)]),
    ]
    gdf = gpd.GeoDataFrame(geometry=polys)

    # Apply to each polygon
    stats_list = [polygon_stats(poly, arr) for poly in gdf.geometry]

    # Convert list of dicts → dataframe → join back
    stats_df = gpd.pd.DataFrame(stats_list)
    gdf = gdf.join(stats_df)

    print(f'gdf is len: {len(gdf)} type: {type(gdf)}')
    print(gdf)

    # plot_img_rois(arr, gdf, stats_df['mask'])

if __name__ == "__main__":
    logger.warning('turning off pd SettingWithCopyWarning')
    pd.options.mode.chained_assignment = None  # default='warn'

    test_intensity()
    
    # fakeData()