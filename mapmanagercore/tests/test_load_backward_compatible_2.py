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

    logger.info('calling points["spineRoi_ch1_count"] -->> will trigger compute')
    values = mmap.points['spineRoi_ch1_count']
    logger.info('spineRoi_ch1_count:')
    print(values)

    spine1 = mmap.points[(20, 1), 'spineLength']
    spine2 = mmap.points[(30, 1), 'spineLength']
    logger.info(f'spine1: {spine1}')
    logger.info(f'spine2: {spine2}')

    allSpineLength = mmap.points['spineLength']
    logger.info(f'allSpineLength: {allSpineLength}')

    return mmap


def test_computation_performance():
    """Test performance of computing various column types."""
    import time
    
    logger.info('=== Performance Test: Computing Columns ===')
    
    # Load the map
    path = get202504_map()
    mmap = MapAnnotations.load_backward_compatible(path)
    logger.info(f'Loaded map with {len(mmap.points)} points and {len(mmap.segments)} segments')
    
    # Test 1: Regular computed column (spineLength)
    logger.info('\n--- Test 1: Regular Computed Column (spineLength) ---')
    start_time = time.time()
    spine_lengths = mmap.points['spineLength']
    end_time = time.time()
    computation_time = end_time - start_time
    logger.info(f'Computed spineLength for {len(spine_lengths)} spines in {computation_time:.4f} seconds')
    logger.info(f'Average time per spine: {computation_time/len(spine_lengths)*1000:.2f} ms')
    logger.info(f'First few values: {spine_lengths.head()}')
    
    # Test 2: Image intensity metric (spineRoi_ch1_mean)
    logger.info('\n--- Test 2: Image Intensity Metric (spineRoi_ch1_mean) ---')
    start_time = time.time()
    roi_means = mmap.points['spineRoi_ch1_mean']
    end_time = time.time()
    computation_time = end_time - start_time
    logger.info(f'Computed spineRoi_ch1_mean for {len(roi_means)} spines in {computation_time:.4f} seconds')
    logger.info(f'Average time per spine: {computation_time/len(roi_means)*1000:.2f} ms')
    logger.info(f'First few values: {roi_means.head()}')
    
    # Test 3: Multiple image metrics at once
    logger.info('\n--- Test 3: Multiple Image Metrics (batch computation) ---')
    image_columns = ['spineRoi_ch1_sum', 'spineRoi_ch1_std', 'spineRoi_ch1_median', 'spineRoi_ch1_cv']
    start_time = time.time()
    image_data = mmap.points[image_columns]
    end_time = time.time()
    computation_time = end_time - start_time
    logger.info(f'Computed {len(image_columns)} image columns for {len(image_data)} spines in {computation_time:.4f} seconds')
    logger.info(f'Average time per column: {computation_time/len(image_columns):.4f} seconds')
    logger.info(f'Average time per spine per column: {computation_time/len(image_columns)/len(image_data)*1000:.2f} ms')
    logger.info(f'Columns computed: {list(image_data.columns)}')
    
    # Test 4: Single spine computation
    logger.info('\n--- Test 4: Single Spine Computation ---')
    start_time = time.time()
    single_spine_length = mmap.points[(20, 1), 'spineLength']
    end_time = time.time()
    computation_time = end_time - start_time
    logger.info(f'Computed spineLength for single spine (20,1) in {computation_time:.4f} seconds')
    logger.info(f'Value: {single_spine_length}')
    
    start_time = time.time()
    single_spine_roi = mmap.points[(20, 1), 'spineRoi_ch1_mean']
    end_time = time.time()
    computation_time = end_time - start_time
    logger.info(f'Computed spineRoi_ch1_mean for single spine (20,1) in {computation_time:.4f} seconds')
    logger.info(f'Value: {single_spine_roi}')
    
    # Test 5: Segment computation
    logger.info('\n--- Test 5: Segment Computation ---')
    start_time = time.time()
    segment_lengths = mmap.segments['length']
    end_time = time.time()
    computation_time = end_time - start_time
    logger.info(f'Computed length for {len(segment_lengths)} segments in {computation_time:.4f} seconds')
    logger.info(f'Average time per segment: {computation_time/len(segment_lengths)*1000:.2f} ms')
    logger.info(f'Values: {segment_lengths}')
    
    logger.info('\n=== Performance Test Complete ===')
    
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
    # old_mmap = test_load_backward_compatible_2()
    
    # Run performance test
    logger.info('\n' + '='*60)
    logger.info('RUNNING PERFORMANCE TEST')
    logger.info('='*60)
    test_computation_performance()
    
    # saveNewMap(old_mmap)
    # loadNewMap()