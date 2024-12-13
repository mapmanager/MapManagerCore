"""A collection of functions to
fetch large data files from MapManagerCore-Data repo using pooch.

See: https://github.com/mapmanager/MapManagerCore-Data

The first fetch will download the file and will take a few seconds.

The next fetch will reload from the local file (no download)?

As a github workflow:
    /home/runner/.cache/pooch
"""

import pooch

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
    urlMap = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u.mmap'
    mapPath = pooch.retrieve(
        url=urlMap,
        known_hash=None,
    )
    return mapPath

def getMultiTimepointMap() -> str:
    urlMap = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/multi_timepoint_map_zip.mmap'
    mapPath = pooch.retrieve(
        url=urlMap,
        known_hash=None,
    )
    return mapPath
