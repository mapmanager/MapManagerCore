from pprint import pprint

try:
    import bioio_base.exceptions
except (ModuleNotFoundError):
    pass

from mapmanagercore.imageImporter import ImageImporter_bioio, ImageImporter_tiff, getImageImporter
import mapmanagercore.data
from mapmanagercore.logger import logger

def test_bioio_exceptions():
    """Test (FileNotFoundError, UnsupportedFileFormatError).
    """
    pathList = [
        # file exists but bad extension
        '/Users/cudmore/Sites/MapManagerCore-Data/data/rr30a_s0u/line_segments.csv',
        # good extension, no file
        '/Users/cudmore/Sites/MapManagerCore-Data/data/ome-tif/no_file.tif'
    ]
    for path in pathList:
        try:
            ii = ImageImporter_bioio(path, loadImgData=True)
        except (bioio_base.exceptions.UnsupportedFileFormatError, FileNotFoundError) as e:
            # logger.error(e)
            continue

def getPathList():
    pathList = [

        mapmanagercore.data.getTiffChannel_2(),  # 3d tif no scale

        mapmanagercore.data.getSampleData('scale-tif'),  # imagej stack with scale
        mapmanagercore.data.getSampleData('max-scale-tif'),  # 1 image slice (with scale)

        mapmanagercore.data.getSampleData('czi'),
        mapmanagercore.data.getSampleData('nd2'),
        # mapmanagercore.data.getSampleData('oir'),  # works but very slow
        
        mapmanagercore.data.getSampleData('ome-tif'),

        # this works but is not usefull (very slow and no feedback, assuming it downloads file first?)
        #'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/t0/rr30a_s0_ch1.tif'
    ]
    return pathList

def test_bioio():
    logger.info('loading lots of files from mapmanagercore-data ...')

    pathList = getPathList()

    for idx, path in enumerate(pathList):
        # logger.info(f'opening file: {path}')
        
        try:
            loadImgData = True
            ii = ImageImporter_bioio(path, loadImgData=loadImgData)
        except (bioio_base.exceptions.UnsupportedFileFormatError, FileNotFoundError) as e:
            # logger.error(e)
            continue

        logger.info(f'  {ii.filename}')
        logger.info(f'   num channels:{ii.numChannels} {ii.channelShape} {ii.physicalPixelSizes}')

        if ii.filename.endswith('rr30a_s0_ch2_imagej_scale.tif') and 'MAX_' not in ii.filename:
            assert ii.numChannels == 1
            assert ii.channelShape == (70, 1024, 1024)
            assert ii.physicalPixelSizes == (1, 0.1500000150000015, 0.1500000150000015)  # scale
        
        elif ii.filename.endswith('MAX_rr30a_s0_ch2_imagej_scale.tif'):
            assert ii.numChannels == 1
            assert ii.channelShape == (1, 1024, 1024)
            assert ii.physicalPixelSizes == (1, 0.1500000150000015, 0.1500000150000015)  # scale
        
        elif ii.filename.endswith('rr30a_s0_ch2.tif'):
            assert ii.numChannels == 1
            assert ii.channelShape == (70, 1024, 1024)
            assert ii.physicalPixelSizes == (1, 1.0, 1.0)  # no scale
        
        # generic sample data, one function getSampleData() with str like 'nd2'
        elif ii.path == mapmanagercore.data.getSampleData('nd2'):
            assert ii.numChannels == 1
            assert ii.channelShape == (29, 1568, 1060)
            assert ii.physicalPixelSizes == (0.2, 0.065, 0.065)
        
        elif ii.filename.endswith('P8_Slice1(moreanterior)LS_NAc1.czi'):
            assert ii.numChannels == 2
            assert ii.channelShape == (7, 784, 784)
            assert ii.physicalPixelSizes == (1.0, 0.0995022551145, 0.0995022551145)
        
        elif ii.filename.endswith('example.ome.tif'):
            assert ii.numChannels == 1
            assert ii.channelShape == (70, 1024, 1024)
            # we need to round xxx because different package versions (numpy, scikit-image, numpy) give different rounding errors !!!
            _expectedPhysicalSize = (1.0, 0.15000001500000149, 0.15000001500000149)
            try:
                assert ii.physicalPixelSizes == _expectedPhysicalSize
            except (AssertionError):
                logger.error(f'expecting physicalPixelSizes {_expectedPhysicalSize} ... but got {ii.physicalPixelSizes}')

        elif ii.filename.endswith('20190320_b_.oir'):
            assert ii.numChannels == 1
            assert ii.channelShape == (151, 512, 512)
            assert ii.physicalPixelSizes == (2.0, 0.497184455521791, 0.497184455521791)
        
        else:
            logger.error(f'DID NOT TEST {ii.filename}')
            continue

        logger.info(f'  !!! file passed:{path}')

def test_zarr_loader():
    from mapmanagercore.lazy_geo_pd_images.loader.zarr import ZarrLoader
    
    # need to refresh mapmanagercore-data with new numpy<2 (abb 20250213)
    # from mapmanagercore.data import getSingleTimepointMap

    # tmp map using parquet
    path = '/Users/cudmore/Desktop/Animal_145_Slice_1_Right.mmap.zip'

    zl = ZarrLoader(path)
    # appendChannels

def test_get_image_importer():
    pathList = getPathList()
    
    for idx, path in enumerate(pathList):
        # logger.info(f'opening file: {path}')
        
        loadImgData = True
        # ii = ImageImporter_tiff(path, loadImgData=loadImgData)
        ii = getImageImporter(path, loadImgData=loadImgData)
        if ii is None:
            logger.error(f'  failed to open file: {path}')
            continue

        logger.info(f'  {ii.filename}')
        logger.info(f'   num channels:{ii.numChannels} {ii.channelShape} {ii.physicalPixelSizes}')

if __name__ == '__main__':

    # test_bioio_exceptions()
    
    #test_bioio()

    # test_zarr_loader()

    test_get_image_importer()