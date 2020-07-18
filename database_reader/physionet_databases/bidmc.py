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
    "BIDMC",
]


class BIDMC(PhysioNetDataBase):
    """ NOT finished,

    BIDMC PPG and Respiration Dataset

    ABOUT bidmc:
    ------------
    1. contains signals and numerics extracted from the much larger MIMIC II matched waveform Database
    2. with manual breath annotations

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. respiration

    References:
    -----------
    [1] https://physionet.org/content/bidmc/1.0.0/
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
        super().__init__(db_name='bidmc', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.data_ext = "dat"
        self.ann_ext = "breath"
        
        self._ls_rec()


    def _ls_rec(self) -> NoReturn:
        """ finished, checked,

        find all records (relative path without file extension),
        and save into `self.all_records` for further use
        """
        try:
            super()._ls_rec()
        except:
            self.all_records = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05', 'bidmc06', 'bidmc07', 'bidmc08', 'bidmc09', 'bidmc10', 'bidmc11', 'bidmc12', 'bidmc13', 'bidmc14', 'bidmc15', 'bidmc16', 'bidmc17', 'bidmc18', 'bidmc19', 'bidmc20', 'bidmc21', 'bidmc22', 'bidmc23', 'bidmc24', 'bidmc25', 'bidmc26', 'bidmc27', 'bidmc28', 'bidmc29', 'bidmc30', 'bidmc31', 'bidmc32', 'bidmc33', 'bidmc34', 'bidmc35', 'bidmc36', 'bidmc37', 'bidmc38', 'bidmc39', 'bidmc40', 'bidmc41', 'bidmc42', 'bidmc43', 'bidmc44', 'bidmc45', 'bidmc46', 'bidmc47', 'bidmc48', 'bidmc49', 'bidmc50', 'bidmc51', 'bidmc52', 'bidmc53']


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
