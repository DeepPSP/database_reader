# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import os
import numpy as np
import pandas as pd
import xmltodict as xtd
from pyedflib import EdfReader
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Iterable, Sequence, NoReturn
from numbers import Real

from database_reader.utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
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
    [2] https://dash.nichd.nih.gov/study/226675
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='nuMoM2b', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
