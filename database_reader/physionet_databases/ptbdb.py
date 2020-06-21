# -*- coding: utf-8 -*-
"""
"""
import os
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from database_reader.utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from database_reader.base import PhysioNetDataBase


__all__ = [
    "PTBDB",
]


class PTBDB(PhysioNetDataBase):
    """ NOT finished,

    PTB (Physikalisch-Technische Bundesanstalt) Diagnostic ECG Database

    ABOUT ptbdb:
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
    [1] https://physionet.org/content/ptbdb/1.0.0/
    [2] https://physionetchallenges.github.io/2020/
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
        super().__init__(db_name='ptbdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        try:
            self.all_records = wfdb.get_record_list('ptbdb')
        except:
            try:
                self.all_records = get_record_list_recursive(self.db_dir, "dat")
            except:
                self.all_records = []
        

    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
