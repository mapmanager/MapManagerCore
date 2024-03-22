from .annotations import Annotations as MapAnnotations
from .loader.imageio import MultiImageLoader
from .loader.mmap import MMapLoader

# added when converting to pip install with setup.py
from ._version import __version__
