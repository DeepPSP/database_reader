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

from database_reader.utils.common import ArrayLike
from database_reader.base import PhysioNetDataBase


__all__ = [
    "PTB_XL",
]


class PTB_XL(PhysioNetDataBase):
    """ NOT finished,

    PTB-XL, a large publicly available electrocardiography dataset

    ABOUT ptb-xl:
    -------------
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
        try:
            # self.all_records = wfdb.get_record_list('ptb-xl')
        except:
            try:
                self.all_records = os.listdir(self.db_path)
                self.all_records = list(set([os.path.splitext(item)[0] for item in self.all_records]))
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
