import json
from typing import Optional

from mapmanagercore.logger import logger

class AnalysisParams():
    def __init__(self, loadJson : str = None):
        
        # self.__version__ = 0.1
        # self.__version__ = 0.1  # switched to dict of dicts
        self.__version__ = 0.2  # 20240508 added anchorPointSearchDistance
        self.__version__ = 0.3  # segmentTracingMaxDistance

        if loadJson is not None:
            self._dict = json.loads(loadJson)
            if self._dict['__version__'] <= self.__version__:
                self._getDefauls()
        else:
            self._getDefauls()
    
    def getDict(self):
        return self._dict

    def getJson(self):
        return json.dumps(self._dict)

    def _getDefauls(self):
        """Get the default dict.
        """
        self._dict = {
            '__version__': self.__version__,
            
            # new spine
            'brightestPathDistance': {
                'defaultValue': 10,
                'currentValue': 10,
                'description': 'points along the tracing to find spine connection (anchor).'
            },

            'channel': {
                'defaultValue': 1,
                'currentValue': 1,
                'description': 'image color channel to find brightest connection of spine.'
            },

            'zSpread': {
                'defaultValue': 3,
                'currentValue': 3,
                'description': 'Number of image slices for max project to find brightest connection of spine.'
            },

            # spine roi
            'roiExtend': {
                'defaultValue': 4,
                'currentValue': 4,
                'description': 'Number of pixels to extend spine head for spine ROI.'
            },

            'roiRadius': {
                'defaultValue': 4,
                'currentValue': 4,
                'description': 'Width of spine ROI.'
            },
            
            # segment
            'segmentRadius': {
                'defaultValue': 4,
                'currentValue': 4,
                'description': 'Radius of segment tracing.'
            },
            
            # The distance 
            'segmentTracingMaxDistance': {
                'defaultValue': 30,
                'currentValue': 30,
                'description': 'Max distance to trace a brightest path with relatively low performance cost.'
            },

            # anchor point search distance
            # 'anchorPointSearchDistance': {
            #     'defaultValue': 10,
            #     'currentValue': 10,
            #     'description': '????.'
            # },

        }

    def __getitem__(self, key) -> Optional[object]:
        """Get the value for a key, return None of KeyError.
        """
        return self.getValue(key)

    def getValue(self, key : str) -> Optional[object]:
        """Get the value for a key, return None of KeyError.
        """
        try:
            return self._dict[key]['currentValue']
        except (KeyError):
            logger.error(f'did not find key "{key}", possible keys are {self._dict.keys()}')

    def setValue(self, key : str, value : object):
        try:
            self._dict[key]['currentValue'] = value
        except (KeyError):
            logger.error(f'did not find key "{key}", possible keys are {self._dict.keys()}')

    def save(self):
        """Save a JSON rep of our _dict to a mm core zarr file.
        """
        pass

    def load(self, path : str):
        """Load JSON from zarr file into our _dict.
        
        Parameters
        ----------
        path : str
            Full path to the zar file.
        """
        pass
