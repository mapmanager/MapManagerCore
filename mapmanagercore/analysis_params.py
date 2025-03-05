import json
import os
from typing import Optional
import zarr
from mapmanagercore.logger import logger

class AnalysisParams():
    """
    """
    def __init__(self, loadedDict : dict = None, path : str = None):
        self.path = path

        # self.__version__ = 0.1
        # self.__version__ = 0.1  # switched to dict of dicts
        self.__version__ = 0.2  # 20240508 added anchorPointSearchDistance
        self.__version__ = 0.3  # segmentTracingMaxDistance
        self.__version__ = 0.4 # abj: added saving and loading
        self.__version__ = 0.5 # abj: adding types and background calculations: Points, Overlap
        # self.__version__ = 0.6 # 20240823 adding type???
        # self.__version__ = 0.6  # abb bumped so we get (backgroundROIGridPoint, backgroundROIGridOverlap)

        self._getDefaults()

        if loadedDict is not None:
            self._loadFromDict(loadedDict)
            # self._dict = json.loads(loadJson)
            # logger.info(f"self._dict['__version__']: {self._dict['__version__'] }")
            # if self._dict['__version__'] < self.__version__:
            #     logger.info("   setting defaults")
            #     self._getDefaults()
    
    # abb this loads from dict
    def _loadFromDict(self, loadedDict : dict):
        """Set values for keys we loaded that we know about.
        """
        # logger.info(f'loadedJson:{loadedDict}')
        # logger.info(f'_loadedDict:{_loadedDict}')
        if isinstance(loadedDict, str):
            loadedDict = json.loads(loadedDict)
        for k,vDict in loadedDict.items():
            if k == '__version__':
                continue
            # logger.info(f'k:{k} v:{v}')
            self.setValue(k, vDict['currentValue'])

    def getDict(self):
        return self._dict

    def printDict(self):
        for k,v in self.getDict().items():
            print(f'{k} {v}')

    def getJson(self, indent : int = 4, excludeVersion : bool = False) -> str:
        # return json.dumps(self._dict, indent=indent)
        if excludeVersion:
            _dict = self._dict.copy()
            _dict.pop('__version__', None)
            return json.dumps(_dict)

        return json.dumps(self._dict)
        
    def setDict(self, newDict):
        """ Applies changes
        """
        self._dict = newDict
    
    def resetDefaults(self) -> dict:
        """ reset and return the default dict
        """
        self._getDefaults()
        return self._dict

    def _getDefaults(self):
        """Get the default dict.
        """
        self._dict = {
            '__version__': self.__version__,

            # new spine
            'brightestPathDistance': {
                'defaultValue': 10,
                'currentValue': 10,
                'description': 'Points along the tracing to find spine connection (anchor).',
                "title": "Brightest Path Distance",
                'type' : "int"
            },

            'channel': {
                'defaultValue': 0,  # 0 based
                'currentValue': 0,
                'description': 'Image color channel to find brightest connection of spine.',
                "title": "Channel",
                'type' : "int"
            },

            'zSpread': {
                'defaultValue': 3,
                'currentValue': 3,
                'description': 'Number of image slices for max project to find brightest connection of spine.',
                "title": "Z Spread",
                'type' : "int"
            },

            # spine roi
            'roiExtend': {
                'defaultValue': 4,
                'currentValue': 4,
                'description': 'Number of pixels to extend spine head for spine ROI.',
                "title": "ROI Extend",
                'type' : "int"
            },

            'roiRadius': {
                'defaultValue': 4,
                'currentValue': 4,
                'description': 'Width of spine ROI.',
                "title": "ROI Radius",
                'type' : "int"
            },

            # segment
            'segmentRadius': {
                'defaultValue': 4,
                'currentValue': 4,
                'description': 'Radius of segment tracing.',
                "title": "Segment Radius",
                'type' : "int"
            },

            # The distance
            'segmentTracingMaxDistance': {
                # 'defaultValue': 90,  # abb was 20
                # 'currentValue': 90,
                'defaultValue': 1000,  # abb was 20
                'currentValue': 1000,
                'description': 'Max distance to trace a brightest path with relatively low performance cost.',
                "title": "Segment Tracing Max Distance",
                'type' : "int"
            },

            # abj
            'maxChannels': {
                # 'defaultValue': 2,
                # 'currentValue': 2,
                'defaultValue': 0,
                'currentValue': 0,
                'description': 'Max number of channels.',
                'type' : "int"
            },

            'backgroundRoiGridPoints': {
                'defaultValue': 5,
                'currentValue': 5,
                'description': 'Number of points used when calculating background ROI. Number of points (n), where grid is n x n',
                "title": "Background ROI Grid Points",
                'type' : "int"
            },

            'backgroundRoiGridOverlap': {
                'defaultValue': 0.1,
                'currentValue': 0.1,
                'description': 'Value that the background grid points are allowed to overlap',
                "title": "Background ROI Grid Overlap",
                'type' : "float"
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

    def getValue(self, key: str) -> Optional[object]:
        """Get the value for a key, return None of KeyError.
        """
        try:
            return self._dict[key]['currentValue']
        except (KeyError):
            logger.error(f'did not find key "{key}", possible keys are {self._dict.keys()}')

    def setValue(self, key : str, value : object) -> Optional[bool]:
        try:
            self._dict[key]['currentValue'] = value
            return True
        except (KeyError):
            logger.error(f'did not find key "{key}", possible keys are {self._dict.keys()}')

    # def getAnalysisParamsFile(self):
    #     userPreferencesFolder = sanpy._util._getUserPreferencesFolder()
    #     optionsFile = pathlib.Path(userPreferencesFolder) / "sanpy_preferences.json"
    #     return optionsFile

    def save(self, externalDict = None):
        """ Save a JSON rep of our _dict to a mm core zarr file.

        Args:
            externalDict: (Optional) - only used when user wants to save changes with an external dictionary
            and not want those changes to be immediately applied to backend. This is used in PMM desktop GUI
        """
        # pass

        logger.info(f"Entering mmc save for analysis params")
        path = self.path 
        # abj
        # save back to zarr file
        if not os.path.isdir(path):
            logger.warning(f'did not find zarr folder: {path}')
            logger.warning('   you may have opened a zar zip, save as a zarr folder and try again')
            return

        zDS = zarr.storage.LocalStore(path, 'w')

        with zDS as store:
            group = zarr.group(store=store)
            # logger.info(f"root.attrs: {root.attrs}")
            # print("root.attrs: ", root.attrs)
            try:
                # _analysisParams_json = group.attrs['analysisParams']  # json str
                # loadedAP = json.loads(_analysisParams_json)
                if externalDict is not None:
                    # currentJson = json.dumps(externalDict, indent=4)
                    currentJson = json.dumps(externalDict)
                else:
                    currentJson = self.getJson()
                group.attrs['analysisParams'] = currentJson
                # group.attrs['__version__'] = self.__version__
                logger.info(f'Saving analysisParams file to {path} ')

            except json.JSONDecodeError as e:
                logger.error(e)
            except TypeError as e:
                logger.error(e)

    def _getDocs(self) -> str:
        """Make self documentation from our dict.
        
        Notes:
            This is not ideal, we really want each key as a row
            and all values like (currentValue, description) as columns
        
            - 5/23 Fixed with transpose
        """
        import pandas as pd
        df = pd.DataFrame(self._dict).transpose()
        return df