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
        # allowPixels=False,
        default=0.6, 
        metadata={
            'description': 'Save metadata version.',
            'doConversion' : False,
            })

    # spines
    # v1
    # brightestPathDistance: int = dataclasses.field(
    #     default=10,  # is currently pixels -> will be in um
    #     metadata={
    #         'description': 'Points along the tracing to find spine connection (anchor).'
    #         })
    # v2
    brightestPathDistance: int = dataclasses.field(
        # allowPixels=True,
        default=10,  # is currently pixels -> will be in um
        metadata={
            'description': 'Points along the tracing to find spine connection (anchor).',
            'doConversion' : True,
            })

    brightestPathChannel: int = dataclasses.field(
        # allowPixels=False,
        default=1, 
        metadata={
            'description': 'Image color channel to find brightest connection of spine.',
            'doConversion' : False
            })
    
    brightestPathZSpread: int = dataclasses.field(
        # allowPixels=False,
        default=3,
        metadata={'description': 'Number of image slices for max project to find brightest connection of spine.',
                    'doConversion' : False,
                })
    
    roiExtend: int = dataclasses.field(
        # allowPixels=True,
        default=4,metadata={'description': 'Number of pixels to extend spine head for spine ROI.',
                            'doConversion' : True,
                            })
    roiRadius: int = dataclasses.field(
        # allowPixels=True,
        default=4,
        metadata={'description': 
                        'Width of spine ROI.',
                        'doConversion' : True,
                })
    
    # segments
    segmentRadius: int = dataclasses.field(
        # allowPixels=True,
        default=4, metadata={'description': 
                            'Radius of segment tracing.',
                            'doConversion' : True,
                            })
    segmentTracingMaxDistance: int = dataclasses.field(
        # allowPixels=True,
        default=90, metadata={'description': 
                            'Max distance to trace a brightest path.',
                            'doConversion' : True,
                            })
    
    backgroundRoiGridPoints: int = dataclasses.field(
        # allowPixels=True,
        default=5, metadata={'description': 
                            'Number of points in grid (nxn) to calculate background ROI.',
                            'doConversion' : False,
                            })

    backgroundRoiGridOverlap: float = dataclasses.field(
        # allowPixels=True,
        default=0.1, metadata={'description': 
                                'Overlap of background grid points.',
                                'doConversion' : True,
                            })

    brightestPathTracing: bool = \
        dataclasses.field(
            # allowPixels=False,
            default=False,
                          metadata=_getMetadata(
                              description='Turn brightest path tracing on and off',
                              title='Brightest Path Tracing',
                              units='',
                              )
        )
