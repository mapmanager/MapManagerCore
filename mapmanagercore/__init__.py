import numpy as np

from .annotations import Annotations as MapAnnotations
from .lazy_geo_pd_images.loader import MultiImageLoader

# abb 20250116

# when importing in gui or scripts
# abb depreciate, use ImageImpoter
IMPORT_FILE_EXTENSIONS = ['.tif',
                          '.tiff',
                          '.nd2',
                          '.czi']

# on load
LOAD_SAVE_EXTENSIONS = ['.mmap', '.zip']

# only accept a limited number of np.dtype
ACCEPTED_DTYPE = [np.uint8, np.int8,
                  np.uint16, np.int16,
                  ]

# auto generated using setuptools_scm
from ._version import __version__