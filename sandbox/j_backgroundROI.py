
import matplotlib.pyplot as plt

from mapmanagercore import MapAnnotations, MultiImageLoader
from mapmanagercore.logger import logger
import mapmanagercore
import mapmanagercore.data

def plot():

    # path_ch1 = mapmanagercore.data.getTiffChannel_1()
    # path_ch1 = "C:\Users\johns\Documents\PyMapManager-Data\PyMapManager-Data\one-timepoint\\rr30a_s0_ch1.tif"
    # path = '\\Users\\johns\\Documents\\GitHub\\MapManagerCore\\data\\rr30a_s0u.mmap'
    path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    map = MapAnnotations.load(path)
    filtered = map.filterPoints(map.points["z"].between(10, 40))
    filtered.points[:, "z"]
    slices = filtered.getPixels(time=0, channel=0)

    fig, ax = plt.subplots(figsize=(10, 10))

    filtered.points["anchorLine"].plot(color='black', ax=ax)
    filtered.points["point"].plot(color='red', marker='o', markersize=2, ax=ax)

    filtered.points["roi"].plot(edgecolor='blue', color=(0,0,0,0), ax=ax)
    filtered.points["roiBg"].plot(edgecolor='red', linestyle='dotted', color=(0,0,0,0), ax=ax)

    slices.plot(ax=ax, vmin=300, vmax=1500, alpha=0.45, cmap='gray')

    # Set x and y limits
    ax.set_xlim(300, 800)
    ax.set_ylim(600, 200)

    plt.show()

def checkBackgroundROI():
    import numpy as np
    import shapely.affinity
    from mapmanagercore.utils import shapeGrid
    from mapmanagercore.lazy_geo_pd_images.loader.base import shapeIndexes
    import geopandas as gp
    import shapely
    import pandas as pd
    from mapmanagercore.analysis_params import AnalysisParams


    
    fig, ax = plt.subplots(figsize=(10, 10))

    path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    # path = '\\Users\\johns\\Documents\\GitHub\\MapManagerCore\\data\\rr30a_s0u.mmap'
    map2 = MapAnnotations.load(path)
    pointID = 5
    # pointID = 37
    pointID = 75
    roi = map2.points[pointID, "roi"]
    z = map2.points[pointID, "z"]

    # TODO: current bug: roi being returned as GeoSeries here, but as Polygon in interactions.py
    logger.error(f'spineId:{pointID} roi is:')
    print(type(roi))
    print(roi)

    # logger.error(f'z:{z} type {type(z)}')
    z = z.iloc[0]
    
    logger.error(f'z.iloc: {z} type {type(z)}')
    
    roi = roi.iloc[0]     # roi should not be a series, extracting geometry 
    logger.info(f"roi.iloc[0] {roi}")
    # overlap = AnalysisParams.getValue("backgroundRoiGridOverlap")
    points = map2.analysisParams.getValue("backgroundRoiGridPoints")
    overlap = map2.analysisParams.getValue("backgroundRoiGridOverlap")
    zSpread = map2.analysisParams.getValue("zSpread")
    channel = map2.analysisParams.getValue('channel')
    # grid = shapeGrid(roi, points=8, overlap=0)
    grid = shapeGrid(roi, points=points, overlap=overlap)
    # logger.info(f"roi {roi} type {type(roi)}")
    # logger.info(f"grid {grid} grid {type(grid)}")

    candidates = gp.GeoSeries(grid.apply(lambda x: shapely.affinity.translate(roi, x["x"], x["y"]), axis=1))
    # pixcels=map2.getShapePixels(candidates, channel=0, zSpread=zSpread, z=map2.points[0, "z"])
    # logger.info(f"candidates {candidates} type {type(candidates)}")

    # had to specify time here? why?
    pixcels=map2.getShapePixels(candidates, channel=channel, zSpread=zSpread, time = 0, z=z)
    # logger.info(f"pixcels {pixcels}")
    
    checkDimmest = pixcels.apply(np.nansum)
    logger.info(f"checkDimmest {checkDimmest}")

    # dimmest = pixcels.apply(np.sum).idxmin()
    dimmest = pixcels.apply(np.nansum).idxmin()

    logger.info(f"dimmest {dimmest} has pixels of {checkDimmest[dimmest]}")
    
    # grid.plot(ax=ax, x="x", y="y", kind="scatter")
    gp.GeoSeries(candidates).plot(ax=ax, facecolor="none", edgecolor="blue")
    gp.GeoSeries([candidates[dimmest]]).plot(ax=ax, facecolor="none", edgecolor="red")
    # grid.iloc[[pixcels.apply(np.sum).idxmin()]].plot(ax=ax, x="x", y="y", color="red", kind="scatter");
    gp.GeoSeries([roi]).plot(ax=ax, facecolor="none", edgecolor="Green")

    zMinus = z - 3
    zPlus = z + 3
    logger.info(f" z value is {z}")
    slices = map2.getPixels(time=0, channel=0, zRange=(zMinus, zPlus))
    xs, ys = shapeIndexes(candidates[dimmest])

    slices.plot(ax=ax, vmin=300, vmax=1500, alpha=1, cmap='CMRmap')
    # pd.DataFrame({"x": xs, "y": ys}).plot(ax=ax, x="x", y="y", kind="scatter", color="red")
    ax.set_xlim(400, 650)
    ax.set_ylim(100, 350)
    # slices._image[xs, ys].sum()
    
    # plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()

def plotAllBackground():

    # mac 
    # path = '/Users/johnsonle/Documents/GitHub/MapManagerCore-Data/data/single_timepoint.mmap'

    # windows 
    path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    # path = '\\Users\\johns\\Documents\\GitHub\\MapManagerCore\\data\\rr30a_s0u.mmap'
    map = MapAnnotations.load(path)

    sessionID = 0
    _sessionMap = map.getTimePoint(sessionID)
    # print(type(_sessionMap))
    df = _sessionMap.points[:]
    temp1 = df["xBackgroundOffset"]

    print("before: ", temp1)
    # print(df.columns.tolist())
    for index in df.index:
        # print(index)
        _sessionMap.snapBackgroundOffset(index)

    fig, ax = plt.subplots(figsize=(5, 5))
    map.points["anchorLine"].plot(color='black', ax=ax)
    map.points["point"].plot(color='red', marker='o', markersize=2, ax=ax)

    map.points["roiHead"].plot(edgecolor='blue', color=(0,0,0,0), ax=ax)
    map.points["roiHeadBg"].plot(edgecolor='blue', linestyle='dotted', color=(0,0,0,0), ax=ax)

    map.points["roiBase"].plot(edgecolor='red', color=(0,0,0,0), ax=ax)
    map.points["roiBaseBg"].plot(edgecolor='red', linestyle='dotted', color=(0,0,0,0), ax=ax)

    slices = map.getPixels(time=0, channel=0, zRange=(18, 36))
    slices.plot(ax=ax, vmin=300, vmax=1500, alpha=0.5, cmap='CMRmap')

    plt.show()
if __name__ == '__main__':
    # plot()
    # checkBackgroundROI()
    plotAllBackground()