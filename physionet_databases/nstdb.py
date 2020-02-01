# -*- coding: utf-8 -*-
import os
import pprint
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real
from ..utils import ArrayLike

from ..base import PhysioNetDataBase


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
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='nstdb', db_path=db_path, **kwargs)
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
