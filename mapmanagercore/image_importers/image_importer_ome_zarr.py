import numpy as np
from bioio import BioImage
import bioio_ome_zarr

from mapmanagercore.image_importers.image_importer_base import ImageImporter_Base

from mapmanagercore.logger import logger

"""
bioio 1.1.0 requires numpy<2.0.0,>=1.21.0,
    but you have numpy 2.1.3 which is incompatible.
bioio 1.1.0 requires zarr<2.18.0,>=2.6.0,
    but you have zarr 2.18.3 which is incompatible.

on pip install bioio, it downgrades to numpy-1.26.4

in general, ome-zarr seems to require numpy-1.26.4

20241204, installed from source
    bioio=1.1.1.dev7+gc293b53

"""
class ImageImporter_Ome_Zarr(ImageImporter_Base):
    def __init__(self, path :str):
        """Load from an ome.zarr file
        
        See here for an online web based loader/viewer
        https://ome.github.io/ome-ngff-validator/

        On that webpage, we can load our export .ome.zarr
        # path = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/single_timepoint.ome.zarr'
        
        To use this we need to run chrome with cors disabled, see:
        https://medium.com/@beligh.hamdi/run-chrome-browser-without-cors-872747142c61

        """
        super().__init__(path)

        # self.bioImage : BioImage = BioImage(path, reader=bioio_ome_zarr.Reader)
        self.bioImage : BioImage = BioImage(path)

        logger.info(f'img.channel_names:{self.bioImage.channel_names}')
        logger.info(f'img.dims.order:{self.bioImage.dims.order}')
        
        # for tiff files, this will usually be wrong
        logger.info(f'img.physical_pixel_sizes.Z: {self.bioImage.physical_pixel_sizes.Z} ')
        logger.info(f'img.physical_pixel_sizes.X: {self.bioImage.physical_pixel_sizes.X} ')
        logger.info(f'img.physical_pixel_sizes.Y: {self.bioImage.physical_pixel_sizes.Y} ')

        # TODO finish this
        self._loadMetadata()

        _t0 = self._loadData_zarr()

        # create a mm core map
        self.loadData(_t0)

    def _loadData_zarr(self, T : int = 0) -> np.ndarray:
        """Pull only a specific chunk in-memory
        returns out-of-memory 4D dask array
        
        T : int
            Pyramid layer
        """

        logger.info('Loading ...')
        
        lazy_t0 = self.bioImage.get_image_dask_data("CZYX", T=T)
        t0 = lazy_t0.compute()  # returns in-memory 4D numpy array
        
        # t0 = self.bioImage.get_image_data('CZYX', T=T)

        logger.info(f'   loaded T:{T} : {t0.shape} {t0.dtype}')
        
        return t0
    
    def _loadMetadata(self):
        """Look for omero meta data.
        """
        img = self.bioImage
        
        # metadata from tifffile is str
        if not isinstance(img.metadata, dict):
            logger.warning(f'metadata:{img.metadata}')
            return
        
        for k,v in img.metadata.items():
        
            if k == 'omero':
                # assuming omero v2 is a dict
                for k2, v2 in v.items():
                    if isinstance(v2, list):
                        for _ch, itemDict in enumerate(v2):
                            logger.info(f'omero k2:{k2} ch:{_ch} is:')
                            for k3, v3 in itemDict.items():
                                # window: {'end': 2675, 'max': 4500, 'min': 0, 'start': 0}
                                interestedIn = ['color', 'window', 'label']
                                if k3 in interestedIn:
                                    logger.info(f'   {k3}: {v3}')

                    # this works but is not needed ???
                    # else:
                        # # unkownKeys = ['id', 'rdefs', 'version']
                        # logger.warning(f'omero did not understand key:"{k2}" value with type: {type(v2)}')
                        # logger.warning(f'   value:{v2}')

            # not interested in this
            # some zarr files will have
            # "_creator": {'name': 'omero-zarr', 'version': '0.4.0'}
            # multiscales : List[dict] that holds scale for each level in pyramid
            # else:
            #     logger.warning(f'ignoring key:"{k}": {v}')

if __name__ == '__main__':
    path = '/Users/cudmore/Dropbox/data/ome-zarr/single_timepoint_v2.ome.zarr'
    path = '/Users/cudmore/Dropbox/data/ome-zarr/single_timepoint_v3.ome.zarr'
    # path = '/Users/cudmore/Dropbox/data/ome-zarr/6001240.zarr'
    
    # from mapmanagercore.data import getTiffChannel_1
    # path = getTiffChannel_1()

    flz = ImageImporter_Ome_Zarr(path)
    map = flz.getMapAnnotations()

    logger.info('AFTER LOAD')
    logger.info(f'map:{map}')

    # save as mmap and open in pyqt
    # flz.saveTo(path='zarrmap')