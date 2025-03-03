[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3111/)
[![tests](https://github.com/mapmanager/MapManagerCore/actions/workflows/test.yml/badge.svg)](https://github.com/mapmanager/MapManagerCore/actions)
[![codecov](https://codecov.io/gh/mapmanager/MapManagerCore/graph/badge.svg?token=M9SO38DYPY)](https://codecov.io/gh/mapmanager/MapManagerCore)
[![OS](https://img.shields.io/badge/OS-Linux|Windows|macOS-blue.svg)]()
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://github.com/mapmanager/MapManagerCore/blob/master/LICENSE)
[![image](http://img.shields.io/pypi/v/mapmanagercore.svg)](https://pypi.python.org/project/mapmanagercore)

# MapManagerCore

MapManagerCore is a Python library that provides the core functionality for MapManager.

An example notebook is located in [examples/example.ipynb](examples/example.ipynb)

## Install

Clone the repo, create a conda environment, install with pip, and run the tests.

    # clone
    git clone https://github.com/mapmanager/MapManagerCore.git
    
    cd MapManagerCore

    # create environment
    conda create -y -n mmc-env python=3.11
    conda activate mmc-env

    # install
    pip install -e '.[tests]'

## Install - Troubleshooting

Remove a conda environment

    conda deactivate
    conda remove -y --name mmc-env --all

    conda info --envs

    # create environment
    conda create -y -n mmc-env python=3.11
    conda activate mmc-env

    # install
    pip install --upgrade --no-cache-dir -e '.[tests]'

Check some important packages

```
pip list | grep -e 'numpy' -e 'pandas' -e 'scikit' -e 'imageio' -e dask -e 'bioio'
```

    bioio                      0.1.dev201+gc34bec4
    bioio-base                 1.0.4
    bioio-czi                  1.0.2
    bioio-nd2                  1.0.0
    bioio-ome-tiff             1.0.1
    bioio-tifffile             1.0.0
    dask                       2025.2.0
    geopandas                  1.0.1
    imageio                    2.37.0
    numpy                      2.2.3
    pandas                     2.2.3
    resource-backed-dask-array 0.1.0
    scikit-image               0.25.1

## Testing

The most important step is to ensure file loaders work

    pytest tests/test_image_importer.py 