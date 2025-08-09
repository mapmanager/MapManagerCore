import numpy as np
import pandas as pd

from mapmanagercore.lazy_geo_pd_images.loader.session_list import SessionList
from mapmanagercore.data import getTiffChannel_1, getTiffChannel_2

def test_make_session_list():
    sl = SessionList()
    
    # s1c1 = np.ndarray((50,512,512), dtype=np.uint8)
    path1 = getTiffChannel_1()
    path2 = getTiffChannel_2()
    
    sl.importSession(path=path1)

def test_session_list():
    pass

if __name__ == '__main__':
    test_session_list()