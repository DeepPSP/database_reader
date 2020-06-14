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
    "PTB_XL",
]


class PTB_XL(PhysioNetDataBase):
    """ NOT finished,

    PTB-XL, a large publicly available electrocardiography dataset

    ABOUT ptb-xl:
    -------------
    1. consists of 21837 clinical 12-lead ECGs from 18885 patients of 10 second length between October 1989 and June 1996
    2. data was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record. The in total 71 different ECG statements conform to the SCP-ECG standard and cover diagnostic, form, and rhythm statements
    3. 

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1.

    References:
    -----------
    [1] https://physionet.org/content/ptb-xl/1.0.1/
    [2] https://physionetchallenges.github.io/2020/
    """
    def __init__(self, db_path:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_path: str, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='ptb-xl', db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
        # wfdb.get_record_list currently not available for this new dataset
        try:
            self.all_records = get_record_list_recursive(self.db_path, "dat")
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
