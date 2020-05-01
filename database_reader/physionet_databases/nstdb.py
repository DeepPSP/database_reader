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
    "NSTDB",
]


class NSTDB(PhysioNetDataBase):
    """ NOT finished,

    MIT-BIH Noise Stress Test Database

    ABOUT qtdb:
    -----------
    to write

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. ecg denoising

    References:
    -----------
    [1] https://physionet.org/content/nstdb/1.0.0/
    """
    def __init__(self, db_path:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        """
        super().__init__(db_name='nstdb', db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
        try:
            self.all_records = wfdb.get_record_list('nstdb')
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
