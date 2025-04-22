import os
from pprint import pprint
import time

from mapmanagercore.lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader
from mapmanagercore.metadata import TimepointMetadata
from mapmanagercore.data import getTiffChannel_1, getTiffChannel_2
from mapmanagercore import MapAnnotations

from mapmanagercore.logger import logger
# import mapmanagercore.metadata

# turn off SettingWithCopyWarning
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

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

    # logger.info('after add spine')
    # print(ma.points[:].columns)

def test_MapAnnotations():
    path1 = getTiffChannel_1()
    path2 = getTiffChannel_2()

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

    # appending a second channel does not creat _ch2 columns for points???
    # if I pre-allocate all _ch1 _ch2 _ch3 in addSchema() then it all get computed
    logger.info('=== after add second channel, print points[:]')
    print(ma.points[['spineRoi_ch1_max', 'spineRoi_ch2_max', 'spineRoi_ch3_max']])
    # print(ma.points[:].columns)

    logger.info('adding one more spine')
    tp = ma.getTimePoint(1)
    _segmentID = 1
    _newSpineID = tp.addSpine(_segmentID, 170, 170, 20)

    logger.info('=== after add new spine, print points[:]')
    print(ma.points[['spineRoi_ch1_max', 'spineRoi_ch2_max', 'spineRoi_ch3_max']])

def test_make_save_map(numTimepoints=1):
    zl = mmMapLoader()

    path1 = getTiffChannel_1()
    path2 = getTiffChannel_2()
    
    for tp in range(numTimepoints):
        # append a new timepoint with image data from a file
        logger.info('appending timepoint 1 from single channel tif')
        _newTimepoint = zl.importTimepoint(path1)
        assert zl.numTimepoints == tp+1

        # append another channel
        logger.info('appending channel to timepoint 1')
        ok = zl.importChannel(path2, timepoint=_newTimepoint)
        assert ok is True
        assert zl.numChannels(_newTimepoint) == 2
        assert 1 in zl.getTimepointMetadata(_newTimepoint).channelKeys

    saveFile = f'zarLoader_{numTimepoints}.mmap'
    savePath = os.path.join('/Users/cudmore/Desktop/sample_mmaps', saveFile)
    zl.saveAs(savePath)

    # logger.info(f'loading saved map: {savePath}')
    # loadedMap = MapAnnotations.load(savePath)

def test_empty_loader(numTimepoint=1):
    logger.info('creating empty zarr loader')
    zl = mmMapLoader()

    path1 = getTiffChannel_1()
    path2 = getTiffChannel_2()
    
    # append a new timepoint with image data from a file
    logger.info('appending timepoint 1 from single channel tif')
    _newTimepoint = zl.importTimepoint(path1)
    logger.info(f'  _newTimepoint:{_newTimepoint}')
    # assert _newTimepoint == 1
    assert zl.numTimepoints == 1
    
    md_tp1 = zl.metadata.getTimepoint(_newTimepoint)
    assert isinstance(md_tp1, TimepointMetadata)

    # print('md_tp1 is:')
    # pprint(md_tp1, sort_dicts=False)

    # zl._printImgSrcs()

    # append another channel
    logger.info('appending channel to timepoint 1')
    ok = zl.importChannel(path2, timepoint=_newTimepoint)
    assert ok is True
    assert zl.numChannels(_newTimepoint) == 2
    # assert 2 in zl.getTimepoint(_tp).channelKeys
    # zl._printImgSrcs()

    # print('before importTimepoint z1 is:')
    # zl.print()

    # if numTimepoint > 1:
    #     # append a second timepoint
    #     logger.info('appending timepoint 2 from single channel tif')
    #     ok = zl.importTimepoint(path1)
    #     assert ok is True
    #     assert zl.numTimepoints == 2

    #     logger.info('after importTimepoint 2nd timepoint, z1 is:')
    #     zl.print()

    # save 1 tp, 2 channels
    # this is runtime metadata before we save

    # print('zl.metadata.asDict() is:')
    # pprint(zl.metadata.asDict())
    
    # return

    savePath = '/Users/cudmore/Desktop/sample_mmaps/zarLoader2.mmap'
    zl.saveAs(savePath)

    # add a 3rd channel to tp 1
    ok = zl.importChannel(path1, timepoint=_newTimepoint)
    assert ok is True
    assert zl.numChannels(_newTimepoint) == 3

    # print('zl.metadata.asDict() after add 3rd channel:')
    # pprint(zl.metadata.asDict())

    # return

    # resave with new channel
    logger.info('resaving with new channel')
    zl.saveAs(savePath)

    # load again
    logger.info(f're-loading saved map: {savePath}')
    loadedMap = MapAnnotations.load(savePath)
    print(loadedMap)


def test_load_zarr() -> mmMapLoader:
    path = '/Users/cudmore/Desktop/sample_mmaps/zarLoader_1.mmap'  # zarLoader2
    logger.info(f'loading:{path}')
    zl = mmMapLoader(path)
    return zl

def test_load_map_annotations():
    # load an mmap like we do in pymapmanager
    path = '/Users/cudmore/Desktop/sample_mmaps/zarLoader_1.mmap'
    mmap = MapAnnotations.load(path)
    print(mmap)

def test_mmmap_image_channel():
    logger.info('test_mmmap_image_channel')
    zl = test_load_zarr()

    # use metadata to traverse
    # _start = time.time()
    for t in zl.metadata.timepointKeys:
        for c in zl.metadata.getTimepoint(t).channelKeys:
            imageChannel = zl.getImageChannel(t, c)
            imageChannel.getSlice(10)
            imageChannel.getSlice(11)
            imageChannel.getSlice(12)
            imageChannel.getSlice(13)
            imageChannel.getSlice(14)

            _vol = imageChannel.getVolume(10, 15)
            print(_vol.shape)

    # _stop = time.time()
    # print(f'took: {_stop - _start}')  # takes ~200 ms to get 20 slices

if __name__ == '__main__':
    
    # test_empty_loader()

    test_MapAnnotations()

    # we can't pass a param, pytest does not have fixture numTimepoints !!!
    # test_make_save_map(1)
    # test_make_save_map(3)

    # load our saved zarr mmap
    # test_load_zarr()

    # test_mmmap_image_channel()

    # load like pymapmanager
    # test_load_map_annotations()