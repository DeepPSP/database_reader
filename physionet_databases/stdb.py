# -*- coding: utf-8 -*-
import os
import pprint
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from utils import ArrayLike
from base import PhysioNetDataBase


__all__ = [
    "STDB",
]


class STDB(PhysioNetDataBase):
    """ NOT finished,

    MIT-BIH ST Change Database

    ABOUT stdb:
    -----------
    1. includes 28 ECG recordings of varying lengths, most of which were recorded during exercise stress tests and which exhibit transient ST depression
    2. the last five records (323 through 327) are excerpts of long-term ECG recordings and exhibit ST elevation
    3. annotation files contain only beat labels; they do not include ST change annotations

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. ST segment

    References:
    -----------
    [1] https://physionet.org/content/stdb/1.0.0/
    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='stdb', db_path=db_path, **kwargs)
        try:
            self.all_records = wfdb.get_record_list('stdb')
        except:
            try:
                self.all_records = os.listdir(self.db_path)
                self.all_records = list(set([os.path.splitext(item)[0] for item in self.all_records]))
            except:
                self.all_records = ['300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327']
        self.all_leads = ['ECG']


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
