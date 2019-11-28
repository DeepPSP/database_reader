# -*- coding: utf-8 -*-
"""
facilities for easy reading of databases from physionet

data from physionet databases can be loaded from its server at use,
or downloaded using `wfdb` easily beforehand
"""

from .apnea_ecg import *
from .bidmc import *
from .cinc2018 import *
from .edb import *
from .ltstdb import *
from .nstdb import *
from .qtdb import *
from .slpdb import *
from .stdb import *


__all__ = [s for s in dir() if not s.startswith('_')]
