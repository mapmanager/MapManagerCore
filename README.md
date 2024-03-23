# MapManagerCore

MapManagerCore is a Python library that provides the core functionality for MapManager.

An example notebook is located under `examples/example.ipynb`.

## Install

Clone the repo, create a conda environment, install with pip, and run the tests.

    git@github.com:mapmanager/MapManagerCore.git
    cd MapManagerCore

    conda create -y -n mmc-env python=3.11
    conda activate mmc-env

    pip install -e '.[dev]'

    python tests/test_base_mutation.py


