from __future__ import absolute_import, division, print_function

from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ""  # use '' for first of series, number for 1 and above
_version_extra = "dev"
# _version_extra = ''  # Uncomment this for full releases

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
    "copt @ git+git://github.com/openopt/copt.git",
    "numpy>=1.11.3",
    "pandas>=0.22.0",
    "scipy>=1.0.0",
    "hyperopt>=0.1.2",
    "scikit-learn>=0.19.1",
    "tqdm>=4.19.4",
    "ipywidgets",
]
EXTRAS_REQUIRE = {"dev": ["flake8", "pytest", "pytest-cov", "coveralls", "pre-commit"]}
ENTRY_POINTS = {}
