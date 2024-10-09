"""A collection of functions to
fetch large data files from MapManagerCore-Data repo using pooch.

See: https://github.com/mapmanager/MapManagerCore-Data

The first fetch will download the file and will take a few seconds.

The next fetch will reload from the local file (no download)?

As a github workflow:
    /home/runner/.cache/pooch
"""

import pooch

def getPointsFile() -> str:
    """Download and get path to points csv.
    """
    urlPoints = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/points.csv'
    pointsPath = pooch.retrieve(
        url=urlPoints,
        known_hash=None
    )
    return pointsPath

def getLinesFile() -> str:
    urlLines = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/line_segments.csv'
    linePath = pooch.retrieve(
        url=urlLines,
        known_hash=None
    )
    return linePath

def getTiffChannel_1() -> str:
    urlCh1 = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/t0/rr30a_s0_ch1.tif'
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

