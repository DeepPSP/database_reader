# -*- coding: utf-8 -*-
import os
import pprint
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from database_reader.utils import ArrayLike
from database_reader.base import PhysioNetDataBase


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
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        """

        """
        super().__init__(db_name='mimic3', db_path=db_path, **kwargs)
        self.all_records = wfdb.get_record_list('mimic3wdb')
        self.freq = 125


    def load_data(self, ):
        """
        """
        raise NotImplementedError


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError
