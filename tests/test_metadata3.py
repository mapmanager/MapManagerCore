import json
from dataclasses import fields, asdict
from pprint import pprint
from typing import List

import numpy as np

from mapmanagercore.metadata.metadata3 import (TimepointMetadata,
                                                         mmMapMetadata,
                                                         ChannelMetadata,
                                                         AnalysisParams,
                                                         ExperimentMetadata)
from mapmanagercore.exceptions import MetadataError

from mapmanagercore.logger import logger

def test_metadata():
    
    tmd = TimepointMetadata()

    # pprint(md, indent=4)

    # test experimentMetadata
    acqDate = tmd.experimentMetadata.getValue('AcqDate')
    assert acqDate==''
    experimentMetadata = tmd.getValue('experimentMetadata')
    experimentMetadata.setValue('AcqDate', 'new_date')
    acqDate = tmd.experimentMetadata.getValue('AcqDate')
    assert acqDate=='new_date'
    # print(f'acqDate:{acqDate}')

    imgData = np.random.randint(low=0, high=2**11, size=(20,512,512), dtype=np.uint16)
    tmd.appendChannel(imgData)  # add default new channels

    assert tmd.numChannels == 1
    # tmd.setChannelName(0, 'my new name!')
    tmd.setChannelProperty(1, 'name', 'my new name!')
    assert tmd.getChannelMetadata(1).getValue('name') == 'my new name!'

    # add a channel with wrong shape
    imgData_bad = np.random.randint(low=0, high=2**11, size=(30,512,512), dtype=np.uint16)
    tmd.appendChannel(imgData_bad)  # add default new channels

    # add a new (good channel)
    imgData2 = np.random.randint(low=12, high=2**8, size=(20,512,512), dtype=np.uint16)
    tmd.appendChannel(imgData2)  # add default new channels

    # print('after appendChannel as dict')
    # pprint(md.asDict())

    # swap a channel
    srcChannelIdx = 0
    dstChannelIdx = 1
    # logger.info('testing swap channels')
    tmd.swapChannels(srcChannelIdx, dstChannelIdx)

    # print('after swapChannels as dict')
    # pprint(tmd.asDict())

    tmd.deleteChannel(5)
    tmd.deleteChannel(1)
    
    # each singleton timepoint needs it's own analysis parameters !!!
    # apDict = md.analysisParameters.getDict()
    # logger.info('ap:')
    # pprint(apDict)

    # iterate channels
    # for metadataContrast in md.metadataContrast:
    #     print(metadataContrast)

    # print(md.channels())

    mdl = mmMapMetadata()

    numTimepoints = 5
    for index in range(numTimepoints):
        tmd = TimepointMetadata()

        imgData = np.random.randint(low=0, high=2**11, size=(20,512,512), dtype=np.uint16)
        newChannelIndex = tmd.appendChannel(imgData)
        assert newChannelIndex == 1
        assert tmd.numChannels == 1
        # tmd.setChannelName(newChannelIndex, f'tp {index} ch {newChannelIndex}')
        tmd.setChannelProperty(newChannelIndex, 'name', f'tp {index} ch {newChannelIndex}')
        imgData = np.random.randint(low=0, high=2**11, size=(20,512,512), dtype=np.uint16)
        newChannelIndex = tmd.appendChannel(imgData)
        assert newChannelIndex == 2
        assert tmd.numChannels == 2
        # tmd.setChannelName(newChannelIndex, f'tp {index} ch {newChannelIndex}')
        tmd.setChannelProperty(newChannelIndex, 'name', f'tp {index} ch {newChannelIndex}')

        mdl.appendTimepoint(tmd)
    
    mdl2 = mmMapMetadata(**asdict(mdl))
    logger.info(mdl2.numItems)

    mdl2.swapTimepoint(1, 2)

    logger.info('after swap 1/2')
    
    logger.info('tp 1')
    pprint(mdl2.getMetadataItem(1))
    logger.info('tp 1')
    pprint(mdl2.getMetadataItem(2))

def test_TimepointMetadata():
    tpmd = TimepointMetadata()

    assert tpmd.numChannels == 0

    _shape = (22,512,512)
    imgData = np.random.randint(low=0, high=2**11, size=_shape, dtype=np.uint16)
    
    newChannelIdx = tpmd.appendChannel(imgData)
    assert tpmd.numChannels == 1
    assert tpmd.shape == _shape

    # delete a bad channel
    _deleted = tpmd.deleteChannel(1000)
    assert not _deleted

    # delete channel 0, not allowed
    channelMetadata = tpmd.deleteChannel(0)
    assert channelMetadata is None

    # add a channel with different shape
    _Badshape = (30,512,512)
    imgDataBad = np.random.randint(low=0, high=2**11, size=_Badshape, dtype=np.uint16)
    newChannelIdx = tpmd.appendChannel(imgDataBad)
    assert newChannelIdx is None

    # add a good channel of different type
    _shape = (22,512,512)
    imgType = np.int8
    imgData = np.random.randint(low=0, high=2**7, size=_shape, dtype=imgType)
    newChannelIdx = tpmd.appendChannel(imgData)
    assert tpmd.numChannels == 2
    assert tpmd.shape == _shape
    
    # set a bad value
    _bad = tpmd.setChannelProperty(channelIdx=None, key='', value=None)
    assert _bad is None

    # set a good value
    _notBad = tpmd.setChannelProperty(channelIdx=1, key='name', value='my new name')
    assert _notBad

    # pprint(tpmd.asDict())

    _dict = tpmd.asDict()
    jsonStr = json.dumps(_dict)
    tpmd2 = TimepointMetadata(**_dict)

    pprint(tpmd2)

def test_analysis_params():
    from mapmanagercore.metadata.analysis_params import AnalysisParams
    
    ap = AnalysisParams()

    # test description in AnalysisParams
    # print(f"segmentRadius: {ap.getDescription('segmentRadius')}")
    # ap.printFields()

    logger.info(f'before set ap.brightestPathDistance:{ap.brightestPathDistance}')
    ap.setValue('brightestPathDistance', 500)
    # could do this, I prefer using API `setValue(key, value)
    # ap.brightestPathDistance = 500
    logger.info(f'after set ap.brightestPathDistance:{ap.brightestPathDistance}')
    # _dict = ap.asDict()
    # pprint(_dict)

    ap.reset_to_defaults()  # in place
    logger.info(f'after reset ap.brightestPathDistance:{ap.brightestPathDistance}')
    # _dict = ap.asDict()
    # pprint(_dict)

    _dictForQtWidget = ap.to_dict_with_metadata()
    logger.info('_dictForQtWidget')
    pprint(_dictForQtWidget)


def test_experimental_metadata():
    #
    # test ExperimentMetadata
    emd = ExperimentMetadata()

    logger.info('initial ExperimentMetadata')
    _dict = emd.asDict()
    pprint(_dict)

    emd.setValue('CellType', 'new cell type')
    logger.info(f'after set emd.CellType:{emd.CellType}')

    emd.reset_to_defaults()
    logger.info(f'after reset emd.CellType:{emd.CellType}')

    _dict = emd.to_dict_with_metadata()
    pprint(_dict)

def test_timepoint_metadata():
    """Test metadata for one timepoint.
    """
    
    tpmd = TimepointMetadata()
    assert tpmd.numChannels == 0

    # append a channel to tpmd, this is on import (we have the whole image volume)
    imgData = np.random.randint(low=0, high=2**11, size=(20,512,512), dtype=np.uint16)
    channelIdx = tpmd.appendChannel(imgData)
    assert channelIdx == 1
    assert tpmd.numChannels == 1

    # set the name of a color channel
    tpmd.setChannelProperty(channelIdx, 'name', '1) my new name')
    assert tpmd.getChannelProperty(channelIdx, 'name') == '1) my new name'

    # add another channel (bad shape)
    imgData = np.random.randint(low=0, high=2**11, size=(10,512,512), dtype=np.uint16)
    badChannelIdx = tpmd.appendChannel(imgData)  # return None on bad shape
    assert badChannelIdx is None

    # add another channel (good shape, different dtype)
    imgData = np.random.randint(low=0, high=2**8, size=(20,512,512), dtype=np.uint8)
    channelIdx = tpmd.appendChannel(imgData)  # ok to add with different dtype
    assert channelIdx == 2

    # swap channels (bad src channel idx)
    _swapped = tpmd.swapChannels(srcChannelIdx=5, dstChannelIdx=0)
    assert _swapped is False

    # swap channels (bad src channel idx)
    _swapped = tpmd.swapChannels(srcChannelIdx=0, dstChannelIdx=2)
    assert _swapped is False
    _swapped = tpmd.swapChannels(srcChannelIdx=2, dstChannelIdx=0)
    assert _swapped is False

    # swap channels (good src and dst)
    _swapped = tpmd.swapChannels(srcChannelIdx=1, dstChannelIdx=2)
    assert _swapped is True

    # check that swapped channel have the correct name/dtype
    name = tpmd.getChannelProperty(1, 'name')
    assert name == '1) my new name'

    name = tpmd.getChannelProperty(2, 'name')
    assert name == 'Untitled'

    # delete channel 5 (error)
    _deleted = tpmd.deleteChannel(channelIdx=5)
    assert _deleted is None

    # delete channel 0 (error)
    _deleted = tpmd.deleteChannel(channelIdx=0)
    assert _deleted is None

    # delete channel 1
    ch1_metadata = tpmd.getChannelMetadata(1)
    _deletedChannel = tpmd.deleteChannel(channelIdx=1)  # return ChannelMetadata
    # pprint(_deletedChannel)
    assert _deletedChannel == ch1_metadata

def test_channel_metadata():
    imgData = np.random.randint(low=0, high=2**11, size=(20,512,512), dtype=np.uint16)
    _max = np.max(imgData)

    cmd = ChannelMetadata()
    cmd._initFromImgData(imgData)

    assert cmd.getValue('maxInt') == _max

def test_mmmap_metadata():
    """mmMapMetadata is a list of TimepointMetadata.
    """
    firstTp = 1
    secondTp = 2

    mdl = mmMapMetadata()
    assert mdl.numTimepoints == 0
    assert mdl._metadataList == {}

    # create single channel timepoint metadata (on user import tiff)
    imgData = np.random.randint(low=0, high=2**11, size=(20,512,512), dtype=np.uint16)
    # append the new timepoint
    newTp = mdl.appendTimepoint(imgData)
    logger.info(f'newTp:{newTp} {type(newTp)}')
    assert newTp == firstTp

    # delete bad timepoint
    _badTp = 3
    try:
        _deletedItem = mdl.deleteTimepoint(_badTp)
        assert _deletedItem is None
    except (MetadataError) as e:
        logger.info(f'expected: {e}')

    # delete good timepoint
    _deletedItem = mdl.deleteTimepoint(newTp)
    assert mdl.numTimepoints == 0
    assert isinstance(_deletedItem, TimepointMetadata)
    assert _deletedItem is not None
    assert _deletedItem.numChannels == 1
    assert _deletedItem.shapeMetadata.xPixels == 512

    # append the new timepoint AGAIN
    _tmp = mdl.appendTimepoint(imgData)
    assert _tmp == firstTp
    assert mdl.numTimepoints == 1

    # append a second timepoint
    imgData2 = np.random.randint(low=0, high=2**8, size=(20,512,512), dtype=np.uint8)
    _tmp = mdl.appendTimepoint(imgData2)
    assert _tmp == secondTp
    assert mdl.numTimepoints == 2

    # print(mdl[0])
    assert mdl.getTimepoint(firstTp) is not None
    assert mdl.getTimepoint(secondTp) is not None
    
    # swap timepoint (bad src)
    try:
        _swapped = mdl.swapTimepoint(srcTimepoint=6, dstTimepoint=0)
        assert _swapped is False
    except (MetadataError) as e:
        logger.info(f'expected: {e}')

    # swap timepoint (bad dst)
    try:
        _swapped = mdl.swapTimepoint(srcTimepoint=1, dstTimepoint=6)
        assert _swapped is False
    except (MetadataError) as e:
        logger.info(f'expected: {e}')

    # check before we swap
    assert mdl.getTimepoint(firstTp).getChannelProperty(0, 'dtype') == 'uint16'
    assert mdl.getTimepoint(secondTp).getChannelProperty(0, 'dtype') == 'uint8'

    # swap timepoint (good)
    mdl.swapTimepoint(srcTimepoint=0, dstTimepoint=1)

    # check that swap worked
    assert mdl.numTimepoints == 2
        
    # pprint(mdl, sort_dicts=False)

    assert mdl.getTimepoint(0).getChannelProperty(0, 'dtype') == 'uint16'
    assert mdl.getTimepoint(1).getChannelProperty(0, 'dtype') == 'uint8'

    # add another channel and test swap
    mdl.getTimepoint(1).appendChannel(imgData)
    assert mdl.getTimepoint(1).numChannels == 2
    mdl.getTimepoint(1).swapChannels(srcChannelIdx=0, dstChannelIdx=1)

    # for idx, tp in enumerate(mdl):
    #     logger.info(f'tp idx:{idx}')
    #     pprint(tp, sort_dicts=False)
    #     for ch in tp:
    #         pprint(ch, sort_dicts=False, indent=4)
 
    mdl.print()

if __name__ == '__main__':
    logger.setLevel('DEBUG')
    
    # test_metadata()

    # test_TimepointMetadata()

    test_analysis_params()
    test_experimental_metadata()

    # test_timepoint_metadata()

    # test_channel_metadata()

    # test_mmmap_metadata()

    # tryGeneric()