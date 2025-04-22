"""A collection of functions to
fetch large data files from MapManagerCore-Data repo using pooch.

See: https://github.com/mapmanager/MapManagerCore-Data

The first fetch will download the file and will take a few seconds.

The next fetch will reload from the local file (no download)?

As a github workflow:
    /home/runner/.cache/pooch
"""

import os
from typing import Optional
import requests

import pooch

from mapmanagercore.logger import logger

def get202504_map() -> str:
    # single_timepoint_202504.mmap
    urlMap = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/202504/single_timepoint_202504.mmap'
    retPath = pooch.retrieve(
        url=urlMap,
        known_hash=None
    )
    return retPath

def getTiffChannel_1() -> str:
    urlCh1 = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/t0/rr30a_s0_ch1.tif'
    # urlCh1 = 'https://download.brainimagelibrary.org/91/2d/912d311d56fe1bce/rr30a/rr30a_s0_ch1.tif'
    ch1Path = pooch.retrieve(
        url=urlCh1,
        known_hash=None
    )
    return ch1Path

def getTiffChannel_2() -> str:
    urlCh2 = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/t0/rr30a_s0_ch2.tif'
    ch2Path = pooch.retrieve(
        url=urlCh2,
        known_hash=None,
    )
    return ch2Path

def getSingleTimepointMap() -> str:
    # TODO put the zip back into mapmanagercore-data
    urlMap = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/single_timepoint.mmap.zip'
    mapPath = pooch.retrieve(
        url=urlMap,
        known_hash=None,
    )
    return mapPath

def getMultiTimepointMap() -> str:
    urlMap = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/multi_timepoint.mmap.zip'
    mapPath = pooch.retrieve(
        url=urlMap,
        known_hash=None,
    )
    return mapPath

# abb 20250204
def getSingleTimepointMap_nd2() -> str:
    urlCh1 = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/olsen/Animal_145_Slice_1_Right.mmap.zip'
    ch1Path = pooch.retrieve(
        url=urlCh1,
        known_hash=None
    )
    return ch1Path

# abb depreciate
def getNd2Channel_1() -> str:
    rootDir = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/samples'
    url = os.path.join(rootDir, 'nd2/Animal_145_Slice_1_Right.nd2')
    path = pooch.retrieve(
        url=url,
        known_hash=None
    )
    return path

# abb 20250204
def getSampleData(item : str) -> Optional[str]:
    rootDir = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/samples'
    
    try:
        if item == 'czi':
            url = os.path.join(rootDir, 'czi/P8_Slice1(moreanterior)LS_NAc1.czi')
        elif item == 'nd2':
            url = os.path.join(rootDir, 'nd2/Animal_145_Slice_1_Right.nd2')
        elif item == 'oir':
            url = os.path.join(rootDir, 'oir/20190320_b_.oir')
        elif item == 'ome-tif':
            url = os.path.join(rootDir, 'ome-tif/example.ome.tif')
        elif item == 'max-scale-tif':
            url = os.path.join(rootDir, 'tiff/MAX_rr30a_s0_ch2_imagej_scale.tif')
        elif item == 'scale-tif':
            url = os.path.join(rootDir, 'tiff/rr30a_s0_ch2_imagej_scale.tif')
        else:
            logger.error(f'did not understand sample data key: "{item}"')
            return
    except (requests.exceptions.HTTPError) as e:
        logger.error(e)
        return

    path = pooch.retrieve(
        url=url,
        known_hash=None
    )
    return path

