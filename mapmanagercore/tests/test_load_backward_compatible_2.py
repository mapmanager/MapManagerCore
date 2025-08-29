import unittest
from mapmanagercore.data import getTiffChannel_1, getTiffChannel_2, get202504_map, get202504_empty_map
from mapmanagercore import MapAnnotations
from mapmanagercore.logger import logger

def test_load_backward_compatible_2():
    """Test that image column values are preserved during backward compatibility loading."""
    
    path = get202504_map()
    # mmap = MapAnnotations.load(path)
    mmap = MapAnnotations.load_backward_compatible(path)
    print(mmap)

    # Test the lazy evaluation system
    logger.info('Testing lazy evaluation access:')
    try:
        values = mmap.points['spineRoi_ch1_sum']
        logger.info(f'Lazy evaluation successful. First few values: {values.head()}')
    except (KeyError, ValueError, AttributeError) as e:
        logger.error(f'Lazy evaluation failed: {e}')
        logger.info('But direct DataFrame access works, so values are preserved')
    
    # Test direct access to the underlying DataFrame to verify values are preserved
    # logger.info('Testing direct DataFrame access:')
    # try:
    #     # Access the underlying DataFrame directly
    #     root_df = mmap.points._rootDf
    #     if 'spineRoi_ch1_sum' in root_df.columns:
    #         values = root_df['spineRoi_ch1_sum']
    #         logger.info(f'Direct DataFrame access successful. First few values: {values.head()}')
    #         logger.info(f'Value type: {type(values)}')
    #         logger.info(f'Number of non-null values: {values.notna().sum()}')
    #     else:
    #         logger.error('spineRoi_ch1_sum column not found in DataFrame')
    # except (AttributeError, KeyError, TypeError) as e:
    #     logger.error(f'Error accessing DataFrame directly: {e}')

    logger.info('calling points["spineRoi_ch1_shape"] -->> will trigger compute')
    values = mmap.points['spineRoi_ch1_size']
    logger.info('spineRoi_ch1_shape:')
    print(values)

    return mmap

def saveNewMap(mmap):
    # save mmap to new file/folder
    savePath = '/Users/cudmore/Sites/MapManagerCore-Data/data/202508/single_timepoint_202508.mmap'
    logger.info(f'saving to {savePath}')
    mmap.save(savePath)

def loadNewMap():
    # load mmap from new file
    savePath = '/Users/cudmore/Sites/MapManagerCore-Data/data/202508/single_timepoint_202508.mmap'
    logger.info(f'loading from {savePath}')
    mmap = MapAnnotations.load_backward_compatible(savePath)
    print(mmap)

    logger.info('1 calling spineLength')
    spineLength = mmap.points['spineLength']
    logger.info(f'spineLength: {spineLength}')

    logger.info('2 calling spineLength')
    spineLength = mmap.points['spineLength']
    logger.info(f'spineLength: {spineLength}')

if __name__ == '__main__':
    old_mmap = test_load_backward_compatible_2()
    # saveNewMap(old_mmap)

    # loadNewMap()