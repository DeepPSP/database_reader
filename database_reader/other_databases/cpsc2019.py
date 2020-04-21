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
    "CPSC2019",
]


class CPSC2019(OtherDataBase):
    """

    The 2nd China Physiological Signal Challenge (CPSC 2019):
    Challenging QRS Detection and Heart Rate Estimation from Single-Lead ECG Recordings

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
    [1] http://2019.icbeb.org/Challenge.html
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """ finished, to be improved,

        Parameters:
        -----------
        db_path: str,
            storage path of the database
        verbose: int, default 2,
        """
        super().__init__(db_name="CPSC2019", db_path=db_path, verbose=verbose, **kwargs)
