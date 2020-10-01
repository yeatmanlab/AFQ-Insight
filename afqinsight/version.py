from __future__ import absolute_import, division, print_function

from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = 1  # use '' for first of series, number for 1 and above
_version_extra = "dev"
# _version_extra = ""  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = ".".join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]

# Description should be a one-liner:
description = (
    "afq-insight: a python library to analyze diffusion "
    "tensor imaging results in the AFQ data format."
)
# Long description will go up on the pypi page
long_description = """
AFQ-Insight
========
AFQ-Insight is a python library designed to analyze diffusion tensor imaging
results in the AFQ data format. This is a work in progress.

License
=======
``afq-insight`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2018, Adam Richie-Halford, Ariel Rokem, University of Washington
"""

NAME = "afqinsight"
MAINTAINER = "Adam Richie-Halford"
MAINTAINER_EMAIL = "richiehalford@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/richford/AFQ-Insight"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Adam Richie-Halford"
AUTHOR_EMAIL = "richiehalford@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {"afqinsight": [pjoin("data", "*", "*")]}
REQUIRES = [
    "bokeh==2.2.0",
    "copt==0.8.4",
    "hyperopt==0.2.4",
    "ipywidgets==7.5.1",
    "matplotlib==3.3.0",
    "numpy==1.19.1",
    "palettable==3.3.0",
    "pandas==1.1.1",
    "scikit-learn==0.23.2",
    "scipy==1.5.2",
    "tables==3.6.1",
    "tqdm==4.48.2",
]
EXTRAS_REQUIRE = {
    "dev": [
        "flake8==3.8.3",
        "pre-commit==2.7.1",
        "pytest-cov==2.10.1",
        "pytest-xdist[psutil]==2.1.0",
        "pytest==6.0.1",
    ]
}
ENTRY_POINTS = {}
