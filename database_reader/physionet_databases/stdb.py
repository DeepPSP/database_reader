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
        super().__init__(db_name='stdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = None  # to check
        self.data_ext = "dat"
        self.ann_ext = "atr"
        
        self._ls_rec()

        self.all_leads = ['ECG']


    def _ls_rec(self, local:bool=True) -> NoReturn:
        """ finished, checked,

        find all records (relative path without file extension),
        and save into `self._all_records` for further use

        Parameters:
        -----------
        local: bool, default True,
            if True, read from local storage, prior to using `wfdb.get_record_list`
        """
        try:
            super()._ls_rec(local=local)
        except:
            self._all_records = ['300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327']


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
