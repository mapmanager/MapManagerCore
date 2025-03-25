import os
from pprint import pprint
import time

from mapmanagercore.lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader, mmMapImageChannel
from mapmanagercore.data import getTiffChannel_1, getTiffChannel_2

from mapmanagercore.logger import logger
import mapmanagercore.metadata

def test_image_map(numTimepoints):
    zl = mmMapLoader()

    path1 = getTiffChannel_1()
    path2 = getTiffChannel_2()
    
    for tp in range(numTimepoints):
        # append a new timepoint with image data from a file
        logger.info('appending timepoint 1 from single channel tif')
        ok = zl.importTimepoint(path1)
        assert ok is True
        assert zl.numTimepoints == tp+1

        # append another channel
        logger.info('appending channel to timepoint 1')
        ok = zl.importChannel(path2, timepoint=tp)
        assert ok is True
        assert zl.numChannels(tp) == 2
        assert 1 in zl.getTimepointMetadata(tp).channelKeys

    saveFile = f'zarrLoader_{numTimepoints}.mmap'
    savePath = os.path.join('/Users/cudmore/Desktop', saveFile)
    zl.saveAs(savePath)

def test_empty_loader(numTimepoint=1):
    logger.info('creating empty zarr loader')
    zl = mmMapLoader()

    path1 = getTiffChannel_1()
    path2 = getTiffChannel_2()
    
    # append a new timepoint with image data from a file
    logger.info('appending timepoint 1 from single channel tif')
    ok = zl.importTimepoint(path1)
    assert ok is True
    assert zl.numTimepoints == 1

    md_tp1 = zl.metadata.getTimepointMetadata(1)
    from mapmanagercore.metadata.metadata3 import TimepointMetadata
    assert isinstance(md_tp1, TimepointMetadata)

    # print('md_tp1 is:')
    # pprint(md_tp1, sort_dicts=False)

    # zl._printImgSrcs()

    # append another channel
    logger.info('appending channel to timepoint 1')
    ok = zl.importChannel(path2, timepoint=1)
    assert ok is True
    assert zl.numChannels(1) == 2
    assert 2 in zl.getTimepointMetadata(1).channelKeys
    # zl._printImgSrcs()

    # print('before importTimepoint z1 is:')
    # zl.print()

    if numTimepoint > 1:
        # append a second timepoint
        logger.info('appending timepoint 2 from single channel tif')
        ok = zl.importTimepoint(path1)
        assert ok is True
        assert zl.numTimepoints == 2

        logger.info('after importTimepoint 2nd timepoint, z1 is:')
        zl.print()

    savePath = '/Users/cudmore/Desktop/zarLoader2.mmap'
    zl.saveAs(savePath)

    # add a 3rd channel to tp 1
    ok = zl.importChannel(path1, timepoint=1)
    assert ok is True
    assert zl.numChannels(1) == 3

    savePath = '/Users/cudmore/Desktop/zarLoader2.mmap'
    zl.saveAs(savePath)

def test_load_zarr() -> mmMapLoader:
    path = '/Users/cudmore/Desktop/zarLoader2.mmap'
    logger.info(f'loading:{path}')
    zl = mmMapLoader(path)
    return zl

def test_mmmap_image_channel():
    logger.info('test_mmmap_image_channel')
    zl = test_load_zarr()

    # timepoint = 1
    # channel = 1
    # imageChannel = mmMapImageChannel(zl, timepoint=timepoint, channel=channel)
    # oneSlice = imageChannel.getSlice(10)
    # print(f'oneSlice:{oneSlice.shape}')

    # use metadata to traverse
    _start = time.time()
    for oneTimepoint in zl.metadata:
        t = oneTimepoint._key
        for oneChannel in oneTimepoint:
            c = oneChannel._key
            #imageChannel = mmMapImageChannel(zl, timepoint=t, channel=c)
            imageChannel = zl.getImageChannel(t, c)
            imageChannel.getSlice(10)
            imageChannel.getSlice(11)
            imageChannel.getSlice(12)
            imageChannel.getSlice(13)
            imageChannel.getSlice(14)

            _vol = imageChannel.getVolume(10, 15)
            print(_vol.shape)

    _stop = time.time()
    print(f'took: {_stop - _start}')  # takes ~200 ms to get 20 slices

if __name__ == '__main__':
    
    # test_empty_loader()

    # test_load_zarr()

    # test_mmmap_image_channel()

    test_image_map(1)
    test_image_map(3)
    