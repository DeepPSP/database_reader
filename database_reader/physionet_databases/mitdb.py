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
    "MITDB",
]


class MITDB(PhysioNetDataBase):
    """ NOT finished,

    MIT-BIH Arrhythmia Database

    ABOUT mitdb:
    ------------
    to write

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1.

    References:
    -----------
    [1] https://physionet.org/content/mitdb/1.0.0/
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
        super().__init__(db_name='mitdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 360
        self.data_ext = "dat"
        self.ann_ext = "atr"

        self._ls_rec()
        

    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
