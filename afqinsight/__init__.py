from __future__ import absolute_import, division, print_function

from . import datasets  # noqa
from . import plot  # noqa
from . import utils  # noqa
from .datasets import *  # noqa
from .insight import *  # noqa
from .transform import *  # noqa
from .sgl import *  # noqa
from .logistic_sgl import *  # noqa
from .version import __version__  # noqa

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
