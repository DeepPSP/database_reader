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
    "PTBDB",
]


class PTBDB(PhysioNetDataBase):
    """ NOT finished,

    PTB (Physikalisch-Technische Bundesanstalt) Diagnostic ECG Database

    ABOUT ptbdb:
    ------------
    1. contains 549 records from 290 subjects, with each subject represented by 1-5 records
    2. aach record includes 15 simultaneously measured signals:
        the conventional 12 leads (i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6),
        together with the 3 Frank lead ECGs (vx, vy, vz)
    3. sampling frequency is 1000 Hz
    4. diagnoses distribution:
        Diagnosis                       # Patients
        Myocardial infarction           148
        Cardiomyopathy/Heart failure	18
        Bundle branch block             15
        Dysrhythmia                     14
        Myocardial hypertrophy          7
        Valvular heart disease          6
        Myocarditis                     4
        Miscellaneous                   4
        Healthy controls                52

    NOTE:
    -----
    1. no subjects numbered 124, 132, 134, or 161
    2. clinical summary (.hea files) is not available for 22 subjects

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
        super().__init__(db_name="ptbdb", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 1000
        self.spacing = 1000/self.freq

        self.rec_ext = "dat"
        self.aux_rec_ext = "xyz"  # the extra 3 Frank lead ECGs
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
