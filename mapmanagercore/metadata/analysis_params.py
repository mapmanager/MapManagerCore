from pprint import pprint

import dataclasses
from dataclasses_json import dataclass_json

from mapmanagercore.metadata._metadata3 import _metadataBase
# from mapmanagercore.exceptions import MetadataError
from mapmanagercore.logger import logger


# trying to generalize creation of metadata dict for dataclass.field()
@dataclasses.dataclass
class _fieldMetadata():
    description: str = ''
    title: str = ''
    units: str = 'um'

def _getMetadata(
        description: str = '',
        title: str = '',
        units: str = 'um',
):
    _dict = {
        'description': description,
        'title': title,
        'units': units,
    }
    fieldMetadata = _fieldMetadata(**_dict)  # unpack dict
    return dataclasses.asdict(fieldMetadata)

@dataclass_json
@dataclasses.dataclass
class AnalysisParams(_metadataBase):
    """Analysis parameters for a single timepoint
    
    Will also be used for all timepoints in a mmap.
    """

    # """Manually increment this when we add to this class."""
    version: float = dataclasses.field(
        default=0.6, # pixels
        # shownValue = # the actual value that is shown in the GUI
        metadata={
            'description': 'Save metadata version.',
            'doConversion' : False,
            })

    # spines
    # TODO: change to brightestPathRange
    brightestPathDistance: int = dataclasses.field(
        default=10, 
        metadata={
            'description': 'Points along the tracing to find spine connection (anchor).',
            'doConversion' : False
            })

    brightestPathChannel: int = dataclasses.field(
        default=1, 
        metadata={
            'description': 'Image color channel to find brightest connection of spine.',
            'doConversion' : False
            })
    
    brightestPathZSpread: int = dataclasses.field(default=3, 
                                                  metadata={'description': 
                                                 'Number of image slices for max project to find brightest connection of spine.',
                                                 'doConversion' : False
                                                    })
    
    # FIXME: figure out how to get shownValue to be in actual units
    # Option 1: have GUI do the math? seems counterintuitive
    # default has to be in actual units, to account for everytime it is reset

    roiExtend: int = dataclasses.field(default=4, 
                                       metadata={'description': 
                                        'Number of pixels to extend spine head for spine ROI.',
                                        'doConversion' : True,
                                        # 'shownValue': 4 # abj: the actual value that is shown in the GUI
                                        }) 
    
    # How to convert this properly
    roiRadius: int = dataclasses.field(default=4, 
                                       metadata={'description': 
                                        'Width of spine ROI.',
                                        'doConversion' : True
                                        }) 
    
    # segments
    segmentRadius: int = dataclasses.field(default=4, 
                                           metadata={'description': 
                                            'Radius of segment tracing.',
                                            'doConversion' : True
                                            }) 
    
    segmentTracingMaxDistance: int = dataclasses.field(default=90, metadata={'description': 
                                                             'Max distance to trace a brightest path.',
                                                            'doConversion' : True
                                                             }) 
    
    backgroundRoiGridPoints: int = dataclasses.field(default=5, metadata={'description': 
                                                             'Number of points in grid (nxn) to calculate background ROI.',
                                                            'doConversion' : False
                                                             })
    backgroundRoiGridOverlap: float = dataclasses.field(default=0.1, metadata={'description': 
                                                             'Shape overlap percentage of background grid points.',
                                                               'doConversion' : False
                                                             }) 
    brightestPathTracing: bool = \
        dataclasses.field(default=False,
                          metadata=_getMetadata(
                              description='Turn brightest path tracing on and off',
                              title='Brightest Path Tracing',
                              units='',
                              )
        )

if __name__ == '__main__':
    _dict = _getMetadata(
        description='xxx',
        title='yyy',
        units='um',
    )
    logger.info('main')
    pprint(_dict)
    pprint(_dict)