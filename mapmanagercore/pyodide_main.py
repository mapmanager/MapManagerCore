import warnings
from .annotations.pyodide import PyodideAnnotations

warnings.filterwarnings("ignore")


async def createAnnotations(path: str) -> PyodideAnnotations:
    """ Create a PyodideAnnotations object from a given path to zar `.mmap` file.
    """
    return PyodideAnnotations.load(path, False)
