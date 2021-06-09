# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive3,
)
from ..base import OtherDataBase


__all__ = [
    "PRCV2021",
]


class PRCV2021(OtherDataBase):
    """

    PRCV 2021 Alzheimer Disease Classification Competition

    ABOUT prcv2021
    --------------
    1. data are sMRI (structural Magnetic Resonance Imaging) data
    2. subjects are divided into three classes:
        - AD (Alzheimer Disease)
        - MCI (Mild Cognitive Impairment)
        - NC (Normal Control)
    3. columns in the stats csv file:
        - new_subject_id: subject id
        - site: 
        - age: age of the subject
        - male: boolean value indicating subject is male (1) or not (0)
        - female: boolean value indicating subject is female (1) or not (0)
        - NC: boolean value indicating subject is of class NC (1) or not (0)
        - MCI: boolean value indicating subject is of class MCI (1) or not (0)
        - AD: boolean value indicating subject is of class AD (1) or not (0)
        - Label: class (map) of the subject, 0 for NC, 1 for MCI, 2 for AD
        - Resolution: sMRI image resolution
        - Noise:
        - Bias:
        - IQR:
        - TIV:
        - CSF:
        - GMV:
        - WMV:
        - Thickness:
        - Thickness_std:

    NOTE
    ----

    ISSUES
    ------
    1. 

    Usage
    -----
    1. alzheimer disease classification

    References
    ----------
    [1] https://competition.huaweicloud.com/information/1000041489/introduction
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs:Any) -> NoReturn:
        """ not finished,

        Parameters
        ----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments
        """
        super().__init__(db_name="PRCV2021", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self._all_records = get_record_list_recursive3(self.db_dir, "^Subject_[\d]{4}\.npy$")
        self.data_ext = "npy"
        self._train_dir = [os.path.dirname(item) for item in self._all_records]
        if len(self._train_dir) != 1:
            raise ValueError("records not in ONE directory")
        self._train_dir = self._train_dir[0]
        try:
            _stats_file = get_record_list_recursive3(self.db_dir, "^train_open\.csv$")[0]
        except:
            raise FileNotFoundError("stats file not found")
        self._stats = pd.read_csv(os.path.join(self.db_dir, f"{_stats_file}.csv"))


    @property
    def df_stats(self):
        """
        """
        return self._stats


    def get_subject_id(self, rec:str) -> int:
        """
        Attach a `subject_id` to the record, in order to facilitate further uses

        Parameters
        ----------
        rec: str,
            record name

        Returns
        -------
        sid: int,
            a `subject_id` attached to the record `rec`
        """
        sid = int(rec.replace("Subject_", ""))
        return sid
