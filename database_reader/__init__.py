# -*- coding: utf-8 -*-
"""
facilities for easy reading of various databases

subpackages:
------------
    physionet_databases
    nsrr_databases
    audio_databases
    image_databases
    other_databases
    utils
"""
import os, sys

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_BASE_DIR)
_IN_SYS_PATH = [p for p in [_BASE_DIR, _PARENT_DIR] if p in sys.path]
if len(_IN_SYS_PATH) == 0:
    sys.path.append(_BASE_DIR)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from .base import *
# from .physionet_databases import *
# from .nsrr_databases import *
# from .audio_databases import *
# from .image_databases import *
# from .other_databases import *


# __all__ = [s for s in dir() if not s.startswith('_')]
