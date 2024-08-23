import json
import os
from typing import Optional
import zarr
from mapmanagercore.logger import logger

class AnalysisParams():
    """
    """
    def __init__(self, loadJson : str = None, path : str = None):
        self.path = path

        # self.__version__ = 0.1
        # self.__version__ = 0.1  # switched to dict of dicts
        self.__version__ = 0.2  # 20240508 added anchorPointSearchDistance
        self.__version__ = 0.3  # segmentTracingMaxDistance
        self.__version__ = 0.4 # abj: added saving and loading
        self.__version__ = 0.5 # abj: adding types and background calculations: Points, Overlap
        # self.__version__ = 0.6 # 20240823 adding type???

        if loadJson is not None:
            self._dict = json.loads(loadJson)
            logger.info(f"self._dict['__version__']: {self._dict['__version__'] }")
            if self._dict['__version__'] < self.__version__:
                logger.info("setting defaults")
                self._getDefaults()
        else:
            self._getDefaults()
    
    def getDict(self):
        return self._dict

    def printDict(self):
        for k,v in self.getDict().items():
            print(f'{k} {v}')

    def getJson(self):
        return json.dumps(self._dict)
    
    def setDict(self, newDict):
        """ Applies changes
        """
        self._dict = newDict
    
    #abj
    def resetDefaults(self):
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
                'description': 'points along the tracing to find spine connection (anchor).',
                'type' : "int"
            },

            'channel': {
                'defaultValue': 1,  # 0 based
                'currentValue': 1,
                'description': 'image color channel to find brightest connection of spine.',
                'type' : "int"
            },

            'zSpread': {
                'defaultValue': 3,
                'currentValue': 3,
                'description': 'Number of image slices for max project to find brightest connection of spine.',
                'type' : "int"
            },

            # spine roi
            'roiExtend': {
                'defaultValue': 4,
                'currentValue': 4,
                'description': 'Number of pixels to extend spine head for spine ROI.',
                'type' : "int"
            },

            'roiRadius': {
                'defaultValue': 4,
                'currentValue': 4,
                'description': 'Width of spine ROI.',
                'type' : "int"
            },
            
            # segment
            'segmentRadius': {
                'defaultValue': 4,
                'currentValue': 4,
                'description': 'Radius of segment tracing.',
                'type' : "int"
            },
            
            # The distance 
            'segmentTracingMaxDistance': {
                'defaultValue': 90,  # abb was 20
                'currentValue': 90,
                'description': 'Max distance to trace a brightest path with relatively low performance cost.',
                'type' : "int"
            },

            'backgroundRoiGridPoints': {
                'defaultValue': 5,
                'currentValue': 5,
                'description': 'Number of points used when calculating background ROI. Number of points (n), where grid is n x n',
                'type' : "int"
            },

            'backgroundRoiGridOverlap': {
                'defaultValue': 0.1,
                'currentValue': 0.1,
                'description': 'Value that the background grid points are allowed to overlap',
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
            print('   error did not find zarr folder', path)
            return

        zDS = zarr.DirectoryStore(path, 'w')

        with zDS as store:
            group = zarr.group(store=store)
            # logger.info(f"root.attrs: {root.attrs}")
            # print("root.attrs: ", root.attrs)
            try:
                # _analysisParams_json = group.attrs['analysisParams']  # json str
                # loadedAP = json.loads(_analysisParams_json)
                if externalDict is not None:
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

    def load(self, path : str):
        """Load JSON from zarr file into our _dict.

        abj: might not be necessary since we load directly from mmap in constructor?
        
        Parameters
        ----------
        path : str
            Full path to the zar file.
        """
        # Maybe have an option for zip?
        if not os.path.isdir(path):
            print('   error did not find zarr folder', path)
            return

        zDS = zarr.DirectoryStore(path, 'r')

        with zDS as store:
            root = zarr.group(store=store)
            # logger.info(f"root.attrs: {root.attrs}")
            try:
                _analysisParams_json = root.attrs['analysisParams']  # json str
                loadedAP = json.loads(_analysisParams_json)
                loadedVersion = loadedAP['__version__'] 
                # logger.info(f"loadedVersion: {loadedVersion}")
                logger.info(f"loading in loadedAP: {loadedAP}")
                if loadedVersion < self.__version__:
                    # use default
                    logger.warning(
                        "  older version found, reverting to current defaults"
                    )
                    logger.warning(
                        f"  loadedVersion:{loadedVersion} currentVersion:{self._version}"
                    )
                    pass
                else:
                    return loadedAP
            except(KeyError):
                print('error, did not find key "analysisParams"')
            except json.JSONDecodeError as e:
                logger.error(e)
            except TypeError as e:
                logger.error(e)

        # logger.info(f"self._dict {self._dict}")
