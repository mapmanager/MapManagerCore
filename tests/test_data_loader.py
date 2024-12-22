# import pandas as pd
import pytest

import mapmanagercore.data
from mapmanagercore import MapAnnotations

def test_load_single_timepoint():
    """Check that we can load files from mapmanagercore.data
    
    This loads from a different repo mapmanagercore-data
    """

    mmapPath = mapmanagercore.data.getSingleTimepointMap()
    print(f'mmapPath:{mmapPath}')

    # check we can load a map
    ok = MapAnnotations.checkFile(mmapPath, verbose=True)
    print(f'ok:{ok}')
    assert ok

    # actually load the map
    map = MapAnnotations.load(mmapPath)
    print(f'map:{map}')
    assert map is not None

def test_load_multi_timepoint():
    mmapPath = mapmanagercore.data.getMultiTimepointMap()
    print(f'mmapPath:{mmapPath}')

    # check we can load a map
    ok = MapAnnotations.checkFile(mmapPath, verbose=False)
    print(f'ok:{ok}')
    assert ok

    # actually load the map
    map = MapAnnotations.load(mmapPath)
    print(f'map:{map}')
    assert map is not None

if __name__ == '__main__':
    test_load_single_timepoint()
    # test_load_multi_timepoint()
    