
import pandas as pd
from mapmanagercore import MapAnnotations #, MultiImageLoader
from mapmanagercore.logger import logger

from mapmanagercore import mmMapLoader
import mapmanagercore.data
from mapmanagercore.lazy_geo_pd_images.loader.base import ImageLoader
# from mapmanagercore.lazy_geo_pd_images.loader.zarr import ZarrLoader
# from ..lazy_geo_pd_images import LazyImagesGeoPandas, ImageLoader

def debugChannelColumns():

    # path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'
    # # path = mapmanagercore.data.getSingleTimepointMap()
    # map = MapAnnotations(path)
    # print(f"map points", map.points)




    # #  TODO: debug mergeFile
    # path = 'C:\\Users\\johns\\Documents\\GitHub\\MapManagerCore-Data\\data\\single_timepoint.mmap'

    # this zarr uses number based folders for images instead of a folder named "images"
    path = '../MapManagerCore-Data/data/202504/single_timepoint_202504.mmap'
    # loader = ZarrLoader(path)

    # path = '/Users/johns/Documents/GitHub/PyMapManager-Data/one-timepoint/rr30a_s0_ch1.tif'
    # loader = mmMapLoader(path)
    # map = MapAnnotations(loader)
    fullMap : MapAnnotations = MapAnnotations.load(path)
    singleTimePoint = fullMap.getTimePoint(1)
    print(f"map points", singleTimePoint.points[:])



    # # newImageLoader2 = mmMapLoader()
    # # newImageLoader.read(path, time=time, channel=channel)
    # path2 = '/Users/johns/Documents/GitHub/PyMapManager-Data/one-timepoint/rr30a_s0_ch2.tif'
    # # newImageLoader2.read(path2, time=0, channel=3, name=None)
    # loader.importChannel(path2, timepoint=0)

    # # loader.merge(newImageLoader2)

    # map2  = MapAnnotations(loader)
    # # # print columns
    # print(f"map points new", map2.points[:])
    # # # map.points[:]



    # loader = MultiImageLoader()

    # path_ch1 = mapmanagercore.data.getTiffChannel_1()

    # loader.read(path_ch1, channel=0)
    # _build : ImageLoader = loader
    # map = MapAnnotations(_build)

    # print("total channels", map._channels())
    # # try and add a second channel to map
    # # we need to add the second channel to LazyImagesGeoPandas._images

    # # -----------------Load 2nd channel ---------------------
    # path_ch2 = mapmanagercore.data.getTiffChannel_2()
    # # loader.read(path_ch2, channel=1)  # ????
    # # _build : ImageLoader = loader.build()
    # # map = MapAnnotations(_build)
    # # -----------------Load 2nd channel ---------------------

    # # print("total channels", map._channels())

    # # need something like
    # # MapAnnotations will need to call its imageloader to read
    # # current problem mapannotations has access to imageloader but not the inherited multiimageloader
    # map.loadInNewChannel(path = path_ch2, channel=1)


def debugChannelWithTifs():
    path = '/Users/johns/Documents/GitHub/PyMapManager-Data/one-timepoint/rr30a_s0_ch1.tif'
    # loader = MultiImageLoader()
    # loader.read(path)
    # map = MapAnnotations(loader)
    
    # print(f"map points", map.points[:])

    loader = mmMapLoader()
    loader.importTimepoint(path)
    
    map = MapAnnotations(loader,
                        lineSegments=pd.DataFrame(),
                        points=pd.DataFrame())

    singleTimePoint = map.getTimePoint(1)
    print(f"map points", singleTimePoint.points[:])

def testingMMAPTif():
    path1 = mapmanagercore.data.getTiffChannel_1()
    path2 = mapmanagercore.data.getTiffChannel_2()

    logger.info('=== creating mmMapLoader')
    mapLoader = mmMapLoader()
    _newTimepoint = mapLoader.importTimepoint(path1)

    logger.info('=== creating MapAnnotations with one channel mmMapLoader')
    ma = MapAnnotations(loader=mapLoader)
    logger.info('print MapAnnotations is:')
    print(ma)
    
    # add a simple segment and spines
    _addSimpleSegmentsAndSpines(ma)

    # good, points only have ch1
    # print(ma.points[:])

    logger.info('=== append a second channel to mmMapLoader')
    ma.loader.importChannel(path2, 1)
    
    logger.info('after adding second channel, ma is:')
    print(ma)

    logger.info('adding same spine segment twice:')
    _addSimpleSegmentsAndSpines(ma)


    # # adding third channel
    # ma.loader.importChannel(path2, 1)

    singleTimePoint = ma.getTimePoint(1)
    print(f"map points", singleTimePoint.points[:])


def _addSimpleSegmentsAndSpines(ma: MapAnnotations):
    tp = ma.getTimePoint(1)
    logger.info('=== adding segment')
    _newSegmentID = tp.newSegment()
    logger.info(f'  _newSegmentID:{_newSegmentID}')

    x = 100
    y = 100
    z = 20
    tp.appendSegmentPoint(_newSegmentID, x, y, z)

    x = 150
    y = 150
    z = 22
    tp.appendSegmentPoint(_newSegmentID, x, y, z)

    # logger.info('after add segment points')
    # print(ma.segments[:])

    logger.info('adding a spine')
    x = 110
    y = 110
    z = 21
    _newSpineID = tp.addSpine(_newSegmentID, x, y, z)

def _testAddChannel():
    """
    """

    timePoint = 1
    path = '../MapManagerCore-Data/data/202504/single_timepoint_202504.mmap'
    fullMap : MapAnnotations = MapAnnotations.load(path)
    singleTimePoint = fullMap.getTimePoint(timePoint)

    # No
    print(f"map no ch 3 points", singleTimePoint.points[:])

    path2 = mapmanagercore.data.getTiffChannel_2()
    logger.info('=== append a second channel to mmMapLoader')
    fullMap.loader.importChannel(path2, timePoint)
    print(f"map points after imports", singleTimePoint.points[:])

    singleTimePoint = fullMap.getTimePoint(timePoint)
    print(f"map points empty ch 3 points", singleTimePoint.points[:])

    # # set channel 3 as active
    fullMap.loader.activateChannel(timePoint, channelIdx=3, activate=False)
    # print(f"map activate ch 3 points", singleTimePoint.points[:])
    print(f"map shouldnt update 1", singleTimePoint.points["denRoiBg_ch3_mean"][133])

    # # _addSimpleSegmentsAndSpines(fullMap)
    index = 133
    singleTimePoint.moveSpine(spineId = index, x=170, y=330, z=20) # different value to force update
    # print(f"map after move", singleTimePoint.points["denRoiBg_ch3_mean"])
    print(f"map shouldnt update 2", singleTimePoint.points["denRoiBg_ch3_mean"][133])

    fullMap.loader.activateChannel(timePoint, channelIdx=3, activate=True)
    print(f"map deactivate ch 3 points", singleTimePoint.points[:])

    # fullMap.restrictChannelCalculationsForPoints(timePoint)
    singleTimePoint = fullMap.getTimePoint(timePoint)
    index = 133
    singleTimePoint.moveSpine(spineId = index, x=171, y=330, z=20) # different value to force update
    # print(f"map after 2nd move", singleTimePoint.points["denRoiBg_ch3_mean"])
    print(f"map should update", singleTimePoint.points["denRoiBg_ch3_mean"][133])
    
    # path = '../MapManagerCore-Data/data/202504/single_timepoint_202504.mmap'
    # loader = mmMapLoader()
    # loader.importTimepoint(path)
    
    # map = MapAnnotations(loader,
    #                     lineSegments=pd.DataFrame(),
    #                     points=pd.DataFrame())

    # fullMap : MapAnnotations = map
    # singleTimePoint = fullMap.getTimePoint(1)
    # print(f"map points", singleTimePoint.points[:])


    # Add channels

    # check columns, should not be calculated

    # update calculated columns

    # check to make sure that is reflected

if __name__ == '__main__':
    # debugChannelColumns()
    # # debugChannelWithTifs()
    # testingMMAPTif()
    _testAddChannel()

# even though we are not calculating ch3
# the code is still looping through those columns