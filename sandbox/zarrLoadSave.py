import io
from io import BytesIO
import os
import json

import pandas as pd
import numpy as np

import zipfile
import zarr

from mapmanagercore import MapAnnotations, MultiImageLoader, MMapLoader

def _getCreateData():
    # some fake (points, lines, images, analysisParams)

    dfPoints = pd.DataFrame()
    dfPoints['x'] = [1,2,3]
    dfPoints['segmentID'] = [0,0,0]

    n = 2000
    dfLines = pd.DataFrame()
    dfLines['segmentID'] = [2] * n
    dfLines['x'] = [3] * n
    dfLines['y'] = [4] * n
    dfLines['z'] = [5] * n

    # fake analysis params dict
    analysisParams = {
        'a': 12,
        'aNother': [1,2,3]
    }
    
    # fake image data
    _images = np.zeros((8,2,80,1024,1024), dtype='uint16')
    _images[:,:,:] += np.random.randint(0, 2048, size=(1024,1024), dtype='uint16')

    return {
        'points': dfPoints,
        'lines': dfLines,
        'images': _images,
        'analysisParams': analysisParams
    }

def _toBytes(df: pd.DataFrame):
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)

def createZarr(path, zipStore=False):
    """Create and save a new zarr.

    This will create an entire zarr file, all datasets including:
        - points
        - lines
        - image
        - analysisparams

    This is used to create a zarr file from scratch.
    It assumes the file does not exist.

    Note:
        image meta data is not included in this example
    """
    print('createZarr()')
    
    _fakeData = _getCreateData()
    dfPoints = _fakeData['points']
    dfLines = _fakeData['lines']
    _images = _fakeData['images']
    analysisParams = _fakeData['analysisParams']
    
    if zipStore:
        path += '.zip'
    
    print('   path:', path)
    if os.path.isdir(path) or os.path.isfile(path):
        print(f'   error: file exists {path}')
        return

    if zipStore:
        cm = zarr.ZipStore(path, mode='w', compression=zipfile.ZIP_STORED)
    else:
        cm = zarr.DirectoryStore(path, 'w')

    with cm as store:
        root = zarr.group(store=store)

        root.create_dataset("points", data=_toBytes(dfPoints),
                             dtype=np.uint8)
        
        root.create_dataset("lineSegments", data=_toBytes(dfLines),
                             dtype=np.uint8)

        print('   images:', _images.shape, _images.dtype, np.min(_images), np.max(_images), np.mean(_images))

        root.create_dataset("images", data=_images,
                                    dtype=_images.dtype)
        
        root.attrs['analysisParams'] = json.dumps(analysisParams)

def updateZarr(path):
    """Update zarr file.

    After the user makes edits, this will save just those items that have been modified.
    TODO: Modified items need to be marked dirty so they know when to save or not.

    This assumes there is an existing zarr file to update.
    """

    print('updateZarr()')

    if not os.path.isdir(path):
        print('   error: did not update, file not found', path)
        return
    
    # some fake modified point data
    n = 1024
    dfPoints = pd.DataFrame()
    dfPoints['spineID'] = range(n)
    dfPoints['segmentID'] = [0] * n

    with zarr.DirectoryStore(path, 'a') as store:
        root = zarr.group(store=store)

        # TODO if points dirty
        pointsDirty = True
        if pointsDirty:
            root.create_dataset("points", data=_toBytes(dfPoints),
                             dtype=np.uint8,
                             overwrite=True)
            print(f'   saved updated points, len:{len(dfPoints)}')

        # TODO if lines dirty
        # TODO if images dirty
        # TODO if analysisParams dirty

def loadZarr(path, zipStore=False):
    """Load zarr.

    Load from an existing zarr file.
    """
    print('loadZarr()')

    if zipStore:
        path += '.zip'
        if not os.path.isfile(path):
            print('   error did not find zarr zip file', path)
            return
    else:
        if not os.path.isdir(path):
            print('   error did not find zarr folder', path)
            return
    
    print('   path:', path)
    
    if zipStore:
        cm = zarr.ZipStore(path, mode='r')
    else:
        cm = zarr.DirectoryStore(path, 'r')

    with cm as store:
        root = zarr.group(store=store)
        
        try:
            points = pd.read_pickle(BytesIO(root["points"][:].tobytes()))
        except(KeyError):
            print('error, did not find key "points"')

        try:
            lineSegments = pd.read_pickle(BytesIO(root["lineSegments"][:].tobytes()))
        except(KeyError):
            print('error, did not find key "lineSegments"')

        _images = root['images']  # json str
        
        # we need to do this here, in class library will just assign
        # self._images (or similar)
        xySlice = _images[0,0,10,:,:]

        try:
            _analysisParams_json = root.attrs['analysisParams']  # json str
            analysisParams = json.loads(_analysisParams_json)
        except(KeyError):
            print('error, did not find key "analysisParams"')

        # TODO: Load image meta data

    print(f'   points:{len(points)}')
    print(f'   lineSegments:{len(lineSegments)}')

    # can't use _images outside scope of 'with'
    # xySlice = _images[0,0,10,:,:]
    print('   images:', _images.shape, _images.dtype, np.min(xySlice), np.max(xySlice), np.mean(xySlice))

    print('   analysisParams:', analysisParams)

if __name__ == '__main__':
    path = 'data/rr30a.mmap'
    
    zipStore = False
    
    createZarr(path, zipStore=zipStore)
    
    updateZarr(path)
    
    loadZarr(path, zipStore=zipStore)