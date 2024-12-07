import numpy as np
import pandas as pd

from mapmanagercore import MapAnnotations, MultiImageLoader
from mapmanagercore.logger import logger

# class FileLoader_Base(MultiImageLoader):
class ImageImporter_Base():
    def __init__(self, path):
        # super().__init__()

        self._path = path
        self._multiImageLoader : MultiImageLoader = MultiImageLoader()
        self._map : MapAnnotations = None

        logger.info(f'path:{path}')

    def getMapAnnotations(self):
        return self._map

    def loadData(self, imgData : np.ndarray, tp : int = 0):
        """Make an mmap from loaded image data.
        
        Parameters:
        ==========
        imgData : np.ndarray
            Image data, first dim is number of channel
        tp : int
            Timepoint to add
        """   

        # self.loader = MultiImageLoader()
        

        _numDims = len(imgData.shape)
        if _numDims < 4:
            # zyx
            _numChannels = 1
        else:
            #CZYX
            _numChannels = imgData.shape[0]

        logger.info(f'adding tp:{tp} numChannels:{_numChannels} shape:{imgData.shape} dtype:{imgData.dtype}')

        for channelIdx in range(_numChannels):
            _channelData = imgData[channelIdx,:] 
            logger.info(f'   adding channelIdx:{channelIdx} tp:{tp} _channelData.shape:{_channelData.shape}')
            logger.info(f'     min:{np.min(_channelData)} max:{np.max(_channelData)} {_channelData.dtype}')
            # add to MultiImageLoader
            self._multiImageLoader.read(_channelData,
                                      channel=channelIdx,
                                      time=tp)
            
        #
        logger.info('building MapAnnotations from MultiImageLoader')
        _build = self._multiImageLoader.build()
        self._map = MapAnnotations(_build,
                            lineSegments=pd.DataFrame(),
                            points=pd.DataFrame())
        
        # self._map.points[:]
        # map._map.segments[:]

    # def saveTo(self, path):
    #     self.save(path)