from typing import Union
import warnings

from .lazy_geo_pd_images.loader.base import ImageLoader
from .annotations.pyodide import PyodideAnnotations

warnings.filterwarnings("ignore")


async def createAnnotations(path: Union[str, None] = None) -> PyodideAnnotations:
    """ Create a PyodideAnnotations object from a given path to zar `.mmap` file.
    """
    if path == None:
        return PyodideAnnotations(ImageLoader())

    return PyodideAnnotations.load(path, False)
