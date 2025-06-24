import os
import numpy as np

from .annotations import Annotations as MapAnnotations
from .lazy_geo_pd_images.loader.mm_map_loader import mmMapLoader

# depreciated
# from .lazy_geo_pd_images.loader import MultiImageLoader

# auto generated using setuptools_scm
from ._version import __version__

# on load
LOAD_SAVE_EXTENSIONS = ['.mmap', '.mmap.zip']

# only accept a limited number of np.dtype
# ACCEPTED_DTYPE = [np.uint8, np.int8,
#                   np.uint16, np.int16,
#                   ]

from mapmanagercore.imageImporter import acceptedExtensions
def canImportPath(path) -> bool:
    """Return True if we can import a file.
    
    This will depend on bioio, default is to just import .tif
    """
    _, filename = os.path.split(path)
    canLoad = False
    for importExt in acceptedExtensions():
        if filename.endswith(importExt):
            canLoad = True
            break
    return canLoad

def canLoadPath(path) -> bool:
    """Return True if we can import a folder (require .mmap)
    """
    canLoad = False
    for importExt in LOAD_SAVE_EXTENSIONS:
        if path.endswith(importExt):
            canLoad = True
            break
    return canLoad
