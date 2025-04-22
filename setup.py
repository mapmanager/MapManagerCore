
import os
from setuptools import setup, find_packages

_thisPath = os.path.abspath(os.path.dirname(__file__))
with open(os.path.abspath(_thisPath+"/README.md")) as f:
    long_description = f.read()

# Check for a configuration flag to disable optional dependencies
pyodide = os.getenv('PYODIDE', '0') == '1'

install_requires = [
    # 'numpy==2.0.0',  # required to use bioio, does this break mapmanagercore?
    # 'numpy>=1.21.0,<2.0.0',
    'numpy',
    'pandas',
    'shapely',
    'geopandas',
    'pyarrow',  # to save/load with parquet
    'scikit-image',
    'zarr>=2.6,<=2.18.7',  # 3.0 has breaking changes
    'async-lru',
    'asyncio',
    'platformdirs',  # to get platform specific App paths
    'plotly',  # needed for colors
    "dataclasses-json",
    'brightest-path-lib',
    'pooch',  # to load data from MapManagerCore-Data repo
    'imagecodecs',
    #'tifffile',

    # image import with bioio
    # 'bioio>=1.2.0',  # this is bleading edge and leads to version problems (worth it)
    'bioio-tifffile',
    'bioio-czi',
    'bioio-nd2',
    'bioio-ome-tiff',
    # # 'bioio-imageio',  # PNG , GIF , & other similar formats seen here
    # # 'bioio-bioformats',  # for oir requires maven/java
]

if pyodide:
    bioioRequirements = ['tifffile']
else:
    bioioRequirements = [
        # install mofified bioio
        #'bioio @ git+ssh://git@github.com/mapmanager/bioio.git',
        'bioio',
        'bioio-tifffile',
        'bioio-czi',
        'bioio-nd2',
        'bioio-ome-tiff',
    ]

testRequirements = [
    'tox',
    'pytest',
    'pytest-cov',
    'flake8',
    'nbformat',
    'matplotlib',
]

devRequirements = [
    'jupyter',
    'ipykernel',  # sometimes required by vs code to run jupyter notebooks
    'matplotlib',
]

docsRequirements = [
    'mkdocs',
    'mkdocs-material',
    'mkdocstrings-python',
]

foundPackages = find_packages(include=['mapmanagercore', 'mapmanagercore.*'])

setup(
    # the package name (on PyPi), still use 'import mapmanagercore'
    name='mapmanagercore',
    # version=VERSION,
    description='MapManagerCore is a Python package that provides the core functionality for MapManager.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mapmanager/MapManagerCore',
    author='Robert H Cudmore',
    author_email='robert.cudmore@gmail.com',
    license='GNU General Public License, Version 3',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
    ],

    # this is cryptic and not well documented
    # but critical to automatically import all sub-modules (folders)
    # package_dir={"" : "mapmanagercore"},

    packages=foundPackages,

    include_package_data=True,  # uses manifest.ini

    use_scm_version=True,
    setup_requires=['setuptools_scm'],

    install_requires=install_requires,

    extras_require={
        'dev': devRequirements,
        'tests': testRequirements,
        'docs': docsRequirements,
        'bioio': bioioRequirements,
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
