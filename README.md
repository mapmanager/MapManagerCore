[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3111/)
[![tests](https://github.com/mapmanager/MapManagerCore/actions/workflows/test.yml/badge.svg)](https://github.com/mapmanager/MapManagerCore/actions)
[![codecov](https://codecov.io/gh/mapmanager/MapManagerCore/graph/badge.svg?token=M9SO38DYPY)](https://codecov.io/gh/mapmanager/MapManagerCore)
[![OS](https://img.shields.io/badge/OS-Linux|Windows|macOS-blue.svg)]()
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://github.com/mapmanager/MapManagerCore/blob/master/LICENSE)
[![image](http://img.shields.io/pypi/v/mapmanagercore.svg)](https://pypi.python.org/project/mapmanagercore)

# MapManagerCore

MapManagerCore is a Python library that provides the core functionality for MapManager.

An example notebook is located under [examples/example.ipynb](examples/example.ipynb)

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

    # run test
    pytest tests


