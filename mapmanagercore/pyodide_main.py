from logging import LogRecord
import logging
from typing import Union
import warnings
from mapmanagercore.logger import logger, ALERT
import js
from .annotations.pyodide import PyodideAnnotations

warnings.filterwarnings("ignore")

class AlertHandler(logging.Handler):
    def emit(self, record: LogRecord):
        if record.levelno >= ALERT:
            js.window.alert(self.format(record))

logger.addHandler(AlertHandler())


def createAnnotations(path: Union[str, None] = None) -> PyodideAnnotations:
    """ Create a PyodideAnnotations object from a given path to zar `.mmap` file.
    """
    return PyodideAnnotations.load(path, False)
