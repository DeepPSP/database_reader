# -*- coding: utf-8 -*-
"""
"""
import os
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

import wfdb
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "MIMIC3",
]


class MIMIC3(PhysioNetDataBase):
    """ NOT Finished,

    MIMIC-III Critical Care Database

    ABOUT mimic3:
    -------------
    1. comprising deidentified health-related data associated with over 4000 patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012
    2. includes information such as demographics, vital sign measurements made at the bedside (~1 data point per hour), laboratory test results, procedures, medications, caregiver notes, imaging reports, and mortality (both in and out of hospital)

    NOTE:
    -----

    ISSUES:
    -------
    ref. [3]

    Usage:
    ------
    1. epidemiology
    2. clinical decision-rule improvement
    3. electronic tool development

    References:
    -----------
    [1] https://mimic.physionet.org/
    [2] https://github.com/MIT-LCP/mimic-code
    [3] https://www.physionet.org/content/mimiciii/1.4/
    [4] https://archive.physionet.org/physiobank/database/mimic3wdb/
    [5] https://archive.physionet.org/physiobank/database/mimic3wdb/matched/
    """
    def __init__(self, db_dir:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="mimic3", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 125
        
        self.data_ext = "dat"
        self.ann_ext = None  # to check
        self._ls_rec(db_name="mimic3wdb")


    def load_data(self, ):
        """
        """
        raise NotImplementedError


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError
