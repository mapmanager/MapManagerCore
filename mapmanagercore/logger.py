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

def setLogLevel(newLogLevel : str = 'INFO'):
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
    Windows: 
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

# Create a custom logger with the name as the module name
logger = logging.getLogger(__name__)


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
