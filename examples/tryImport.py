import sys

import matplotlib.pyplot as plt

from mapmanagercore import MapAnnotations, MultiImageLoader, MMapLoader

from mapmanagercore.logger import logger
from mapmanagercore.logger import setLogLevel

setLogLevel()

def tryIt():
    if 0:
        # Create an image loader
        loader = MultiImageLoader(
            lineSegments="../data/rr30a_s0u/line_segments.csv",
            points="../data/rr30a_s0u/points.csv",
            metadata="../data/rr30a_s0u/metadata.json",)

        # add image channels to the loader
        loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch1.tif", channel=0)
        loader.read("../data/rr30a_s0u/t0/rr30a_s0_ch2.tif", channel=1)

        # Create the annotation map
        map = MapAnnotations(loader)

        # save the annotation map
        print('saving ../data/cudmoreMap.mmap')
        map.save("../data/cudmoreMap.mmap")

        return

    map = MapAnnotations(MMapLoader("../data/cudmoreMap.mmap").cached())

    # print(map._table().head())

    # cols = map.columns
    # print(f'cols:{cols}')

    # each columnsAttribute is like
    # {'categorical': True, 'index': False, 'plot': True, 'title': 'Segment ID', 'column': 'segmentID'}
    columnsAttributes = map.columnsAttributes
    # print(f'columnsAttributes:')
    # for columnsAttribute in columnsAttributes:
    #     print(f'   {columnsAttribute}')

    # from mapmanagercore.config import Spine
    # aSpine = Spine.defaults()
    # print(aSpine)
    # segmentIdType = type(aSpine['segmentID'])
    # print(f'segmentIdType:{segmentIdType}')

    # sys.exit(1)

    # print("map['z']")
    spineIDs = map.spineID()
    print('len(spineIDs):', len(spineIDs))

    # mapmanagercore.annotations.base.layers.AnnotationsLayers
    # filtered = map[spineIDs]

    allSpines = map[ map['t']==0 ]  # mapmanagercore.annotations.base.layers.AnnotationsLayers
    allSpines = allSpines[:]  # geopandas.geodataframe.GeoDataFrame
    print(allSpines.columns)
    print(allSpines)

    # get a subset of columns
    cols = ['x', 'y', 'z', 'segmentID', 'note', 'userType', 'spineLength']
    df = map[cols]  # geopandas.geodataframe.GeoDataFrame
    print(df)

    #
    # map.segments['segment'] is geopandas.geoseries.GeoSeries
    xyz = map.segments["segment"].get_coordinates(include_z=True)  # pd.DataFrame
    # print(map.segments.columns)  # columns is ['segment', 'segmentLeft', 'segmentRight']

    # get spine line
    anchorDf = map['anchors'].get_coordinates(include_z=True)
    print('ananchorDf')
    print(anchorDf)
    """
                x      y   z
    spineID                  
    0        425.0  225.4 NaN
    0        431.0  239.0 NaN
    1        378.0  236.0 NaN
    1        382.0  250.0 NaN
    """

if __name__ == '__main__':
    tryIt()