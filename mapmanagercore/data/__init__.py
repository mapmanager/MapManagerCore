import pooch

def getPointsFile():
    """Download and get path to points csv.
    """
    urlPoints = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/points.csv'
    # urlPoints = 'https://raw.githubusercontent.com/mapmanager/MapManagerCore-Data/main/data/rr30a_s0u/points.csv'
    pointsPath = pooch.retrieve(
        url=urlPoints,
        known_hash=None  #'39f8ed2efc8212dd606a721f5a952864ea64ad4f1b6ba9816a3a91cc5471808c',
    )
    return pointsPath

def getLinesFile():
    urlLines = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/line_segments.csv'
    # urlLines = 'https://raw.githubusercontent.com/mapmanager/MapManagerCore-Data/main/data/rr30a_s0u/line_segments.csv'
    linePath = pooch.retrieve(
        url=urlLines,
        known_hash=None  #'bcb3c7d6e1df2f75b9eb69dedfc9cc4b5cb688f1ae4587030b84c0212b7bb727',
    )
    return linePath

def getTiffChannel_1():
    urlCh1 = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/t0/rr30a_s0_ch1.tif'
    # urlCh1 = 'https://github.com/mapmanager/MapManagerCore-Data/blob/main/data/rr30a_s0u/t0/rr30a_s0_ch1.tif'
    ch1Path = pooch.retrieve(
        url=urlCh1,
        known_hash=None  #'d72194e50130f46b7d7cb7cfdb8a425d9da995236c03c4d940d5d60c3e42c35e',
    )
    return ch1Path

def getTiffChannel_2():
    urlCh2 = 'https://github.com/mapmanager/MapManagerCore-Data/raw/main/data/rr30a_s0u/t0/rr30a_s0_ch2.tif'
    # urlCh2 = 'https://github.com/mapmanager/MapManagerCore-Data/blob/main/data/rr30a_s0u/t0/rr30a_s0_ch2.tif'
    ch2Path = pooch.retrieve(
        url=urlCh2,
        known_hash=None,
    )
    return ch2Path


