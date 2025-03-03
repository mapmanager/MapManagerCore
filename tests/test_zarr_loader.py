from pprint import pprint

from mapmanagercore.lazy_geo_pd_images.loader.zarr import ZarrLoader
from mapmanagercore.logger import logger
from mapmanagercore.data import getTiffChannel_1, getTiffChannel_2, getNd2Channel_1

def test_zarr_loader():

    # empty zarr loader
    zl = ZarrLoader()

    print(zl)

    # default zarr loader makes [{}], we just want []
    print(zl._imagesSrcs)  # [{}]
    zl._imagesSrcs = []

    path1 = getTiffChannel_1()
    path2 = getTiffChannel_2()
    
    # error, bad timepoint
    timepoint = 0
    _ok = zl.appendChannels_ii(path1, timepoint=timepoint)
    assert _ok is None

    # error, file not found
    _ok = zl.appendTimepoint_ii(path='nofile.txt', verbose=False)
    assert _ok is None

    # todo: add good file with bad extension

    # append a new timepoint
    _ok = zl.appendTimepoint_ii(path1, verbose=True)
    assert _ok is True
    assert zl._numTimepoints_ii == 1
    assert zl._metadata3.numTimepoints == 1

    # append a new timepoint
    _ok = zl.appendTimepoint_ii(path1, verbose=True)
    assert _ok is True
    assert zl._numTimepoints_ii == 2
    assert zl._metadata3.numTimepoints == 2

    # logger.info('zl._metadata3 is:')
    # pprint(zl._metadata3)

    # append channel to timpoint 1
    timepoint = 1
    _ok = zl.appendChannels_ii(path2, timepoint=timepoint)
    assert _ok is True
    assert zl._numChannels_ii(timepoint) == 2

    # append channel to timpoint 1
    timepoint = 0
    _ok = zl.appendChannels_ii(path2, timepoint=timepoint)
    assert _ok is True
    assert zl._numChannels_ii(timepoint) == 2

    # bad channel shape
    nd2Path = getNd2Channel_1()
    timepoint = 1
    _ok = zl.appendChannels_ii(nd2Path, timepoint=timepoint)
    assert _ok is not True
    assert zl._numChannels_ii(timepoint) == 2  # still 2, append channel failed

if __name__ == '__main__':
    test_zarr_loader()