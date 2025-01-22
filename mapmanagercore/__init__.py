from .annotations import Annotations as MapAnnotations
from .lazy_geo_pd_images.loader import MultiImageLoader

# abb 20250116
IMPORT_FILE_EXTENSIONS = ['.tif']
LOAD_SAVE_EXTENSIONS = ['.mmap', '.zip']

# auto generated using setuptools_scm
from ._version import __version__