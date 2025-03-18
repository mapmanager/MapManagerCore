from pprint import pprint

from mapmanagercore.lazy_geo_pd_images.loader.zarrloader import ZarrLoader2
from mapmanagercore.data import getTiffChannel_1, getTiffChannel_2

from mapmanagercore.logger import logger
import mapmanagercore.metadata

def test_empty_loader():
    logger.info('creating empty zarr loader')
    zl = ZarrLoader2()

    path1 = getTiffChannel_1()
    path2 = getTiffChannel_2()
    
    # append a new timepoint with image data from a file
    logger.info('appending timepoint from single channel tif')
    ok = zl.appendTimepoint(path1)
    assert ok is True
    assert zl.numTimepoints == 1

    md_tp1 = zl.metadata.getTimepointMetadata(1)
    from mapmanagercore.metadata.metadata3 import TimepointMetadata
    assert isinstance(md_tp1, TimepointMetadata)
    pprint(md_tp1, sort_dicts=False)

    # zl._printImgSrcs()

    # append another channel
    logger.info('appending channel to timepoint')
    ok = zl.appendChannels(path2, timepoint=1)
    assert ok is True
    assert zl.numChannels(1) == 2

    # zl._printImgSrcs()

    # append a second timepoint
    ok = zl.appendTimepoint(path1)
    assert ok is True
    assert zl.numTimepoints == 2

    zl._printImgSrcs()

    savePath = '/Users/cudmore/Desktop/zarLoader2.mmap'
    zl.saveAs(savePath)

    # add a 3rd channel to tp 1
    ok = zl.appendChannels(path1, timepoint=1)
    assert ok is True
    assert zl.numChannels(1) == 3

    savePath = '/Users/cudmore/Desktop/zarLoader2.mmap'
    zl.saveAs(savePath)

def test_load_zarr():
    path = '/Users/cudmore/Desktop/zarLoader2.mmap'
    logger.info(f'loading:{path}')
    zl = ZarrLoader2(path)

if __name__ == '__main__':
    
    # test_empty_loader()

    test_load_zarr()