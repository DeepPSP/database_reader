# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import os
import numpy as np
import pandas as pd
import pprint
import xmltodict as xtd
from pyedflib import EdfReader
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Iterable, Sequence, NoReturn
from numbers import Real
from ..utils import ArrayLike
from utils.utils_interval import intervals_union

from ..base import NSRRDataBase


__all__ = [
    "MESA",
]


class MESA(NSRRDataBase):
    """
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """
        """
        super().__init__(db_name='MESA', db_path=db_path, verbose=verbose, **kwargs)
    