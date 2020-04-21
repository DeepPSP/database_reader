# -*- coding: utf-8 -*-
"""
"""
import io
import os
import pprint
import wfdb
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, NoReturn
from numbers import Real

from database_reader.utils import ArrayLike
from database_reader.base import OtherDataBase


__all__ = [
    "CPSC2020",
]


class CPSC2020(OtherDataBase):
    """

    The 3rd China Physiological Signal Challenge 2020:
    Searching for Premature Ventricular Contraction and Supraventricular Premature Beat from Long-term ECGs

    ABOUT CPSC2019:
    ---------------

    NOTE:
    -----    

    ISSUES:
    -------

    Usage:
    ------

    References:
    -----------
    [1] http://www.icbeb.org/CPSC2020.html
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """ finished, to be improved,

        Parameters:
        -----------
        db_path: str,
            storage path of the database
        verbose: int, default 2,
        """
        super().__init__(db_name="CPSC2020", db_path=db_path, verbose=verbose, **kwargs)
