
import os
from setuptools import setup, find_packages

_thisPath = os.path.abspath(os.path.dirname(__file__))
with open(os.path.abspath(_thisPath+"/README.md")) as f:
    long_description = f.read()

install_requires = [
    'async-lru',
    'asyncio',
    'tifffile',
    'imagecodecs',  # required for compression
    'plotly',  # required for compression
    'numpy',
    'pandas',
    'shapely',
    'geopandas==1.0.0a1',
    'scikit-image',
    'zarr'
]

testRequirements = [
    'tox',
    'pytest',
    'pytest-cov',
    'flake8',
]

devRequirements = [
    'jupyter',
    'ipykernel', # sometimes required by vs code to run jupyter notebooks
    'matplotlib',
]

foundPackages = find_packages(include=['mapmanagercore', 'mapmanagercore.*'])

setup(
    name='mapmanagercore',  # the package name (on PyPi), still use 'import mapmanagercore'
    #version=VERSION,
    description='MapManagerCore is a Python library that provides the core functionality for MapManager.',
    long_description=long_description,
    long_description_content_type = 'text/markdown',
    # TODO: assign to url of mapmanagercore
    # url='http://github.com/cudmore/SanPy',
    # how do we specify multiple authors?
    author='Robert H Cudmore',
    author_email='robert.cudmore@gmail.com',
    license='GNU General Public License, Version 3',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Research Science Apps.',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
    ],

    # this is criptic and not well documented
    # but critical to automatically import all sub-modules (folders)
    # package_dir={"" : "mapmanagercore"},

    packages=foundPackages,

    include_package_data=True,  # uses manifest.in

    use_scm_version=True,
    setup_requires=['setuptools_scm'],

    install_requires=install_requires,

    extras_require={
        'dev': devRequirements,
        'tests': testRequirements,
    },

    python_requires=">=3.11",

    # we do not have an entry point
    # if we did, it would look like this
    # entry_points={
    #     'console_scripts': [
    #         'sanpy=sanpy.interface.sanpy_app:main',
    #     ]
    # },
)