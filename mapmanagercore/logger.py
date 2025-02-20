"""Log to both terminal and a file.

Usage
-----
from mapmanagercore.logger import logger

logger.info('info log')
logger.warning('warning log')
logger.error('error log')
"""

import logging
import os
import sys

from logging.handlers import RotatingFileHandler
from typing import Concatenate, Union

def setLogLevel(newLogLevel : str = 'DEBUG'):
    """Set the global logging level.
    
    Can update this during runtime and all logs will follow the level.
    """
    logger = logging.getLogger(__name__)

    print(f'setLogLevel() newLogLevel "{newLogLevel}"')

    if newLogLevel == 'DEBUG':
        logLevel = logging.DEBUG
    elif newLogLevel == 'INFO':
        logLevel = logging.INFO
    elif newLogLevel == 'WARNING':
        logLevel = logging.WARNING
    elif newLogLevel == 'ERROR':
        logLevel = logging.ERROR
    elif newLogLevel == 'CRITICAL':
        logLevel = logging.CRITICAL
    else:
        errStr  = f'did not understand new log level "{newLogLevel}"'
        print('   ', errStr)
        logger.error(errStr)
        return
    
    logger.setLevel(logLevel)

def getLoggerFilePath():
    """All MapManager code will log to the same place including:
     - MapManagerCore
     - MapManagerQt

    Notes
    -----
    Mac: /Users/cudmore/Library/Application Support/MapManager
    Linux: /share/MapManager
    Windows: /Users/johns/AppData/Local/MapManager
    """
    appName = 'MapManager'
    try:
        from platformdirs import user_data_dir  # to get log path
        appDir = user_data_dir(appName)
        if not os.path.isdir(appDir):
            os.makedirs(appDir, exist_ok=True)
    except ImportError:
        appDir = os.path.join(os.path.expanduser('~'), '.mapmanager')
        os.makedirs(appDir, exist_ok=True)

    logFilePath = os.path.join(appDir, 'mapmanager.log')
    return logFilePath

setLogLevel()

class AlertLogger(logging.Logger):
    """A logger that can alert the user."""

    def alert(self, message: str):
        """ Log a message with level ALERT on this logger.
        In pyodide this will pop up an alert box.
        
        Args:
            message (str): The message to log or show the user in the alert box.
        """
        pass

# Create a custom logger with the name as the module name
logger: AlertLogger = logging.getLogger(__name__)
ALERT = logging.ERROR + 1

# Monkey patch the alert method
logging.addLevelName(ALERT, 'ALERT')

def alert(self: logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(ALERT):
        self._log(ALERT, message, args, **kwargs)

logging.Logger.alert = alert

handler = logging.StreamHandler(sys.stdout)

logFilePath = getLoggerFilePath()
f_handler = RotatingFileHandler(logFilePath, maxBytes=2e6, backupCount=1)

# I want the class name of the caller
# this gives us the filename _lologger()
# [%(name)s()]
# [%(module)s()]
#formatter = logging.Formatter('%(levelname)7s - [%(module)s()] %(filename)s %(funcName)s() line:%(lineno)d -- %(message)s')
formatter = logging.Formatter('%(levelname)7s - %(filename)s %(funcName)s() line:%(lineno)d -- %(message)s')
handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(f_handler)
