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

from database_reader.utils import ArrayLike
from database_reader.utils.utils_universal import intervals_union
from database_reader.base import NSRRDataBase


__all__ = [
    "nuMoM2b",
]


class nuMoM2b(NSRRDataBase):
    """

    Nulliparous Pregnancy Outcomes Study: Monitoring Mothers-to-Be (nuMoM2b)

    ABOUT nuMoM2b:
    --------------

    NOTE:
    -----
    1. 

    ISSUES:
    -------

    Usage:
    ------
    1. sleep analysis, especially for pregnant women

    References:
    -----------
    [1] https://sleepdata.org/datasets/numom2b
    """
    def __init__(self, db_path:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        """
        super().__init__(db_name='nuMoM2b', db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
