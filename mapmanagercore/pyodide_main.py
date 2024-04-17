import warnings
from .annotations.pyodide import PyodideAnnotations

warnings.filterwarnings("ignore")

async def createAnnotations(path: str) -> PyodideAnnotations:
    return await PyodideAnnotations.load(path)
