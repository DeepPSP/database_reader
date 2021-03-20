# -*- coding: utf-8 -*-
"""
"""
import os
import random
import math
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, Dict, Sequence, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
from scipy.io import loadmat
from easydict import EasyDict as ED

from ..utils.common import (
    ArrayLike,
    DEFAULT_FIG_SIZE_PER_SEC,
)
from ..base import OtherDataBase


__all__ = [
    "CPSC2021",
]


class CPSC2021(OtherDataBase):
    """

    The 4th China Physiological Signal Challenge 2021:
    Paroxysmal Atrial Fibrillation Events Detection from Dynamic ECG Recordings

    ABOUT CPSC2021:
    ---------------
    1. source ECG data are recorded from 12-lead Holter or 3-lead wearable ECG monitoring devices
    2. dataset provides variable-length ECG fragments extracted from lead I and lead II of the long-term source ECG data, each sampled at 200 Hz
    3. AF event is limited to be no less than 5 heart beats
    4. training set in the 1st stage consists of 716 records, extracted from the Holter records from 12 AF patients and 42 non-AF patients (usually including other abnormal and normal rhythms)
    5. test set comprises data from the same source as the training set as well as DIFFERENT data source, which are NOT to be released at any point
    6. annotations are standardized according to PhysioBank Annotations (Ref. [2] or PhysioNetDataBase.helper), and include the beat annotations (R peak location and beat type), the rhythm annotations (rhythm change flag and rhythm type) and the diagnosis of the global rhythm
    7. 

    NOTE:
    -----
    1. 

    ISSUES:
    -------
    1. 

    TODO:
    -----
    1. 

    Usage:
    ------
    1. 

    References:
    -----------
    [1] http://www.icbeb.org/CPSC2021
    [2] https://archive.physionet.org/physiobank/annotations.shtml
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """ NOT finished,

        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="CPSC2021", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)

        self.fs = 200
        self.spacing = 1000/self.fs
        # self.rec_ext = ".mat"
        # self.ann_ext = ".mat"

        # self.nb_records = 10
        # self._all_records = [f"A{i:02d}" for i in range(1,1+self.nb_records)]
        # self._all_annotations = [f"R{i:02d}" for i in range(1,1+self.nb_records)]
        # # self.all_references = self.all_annotations
        # self.rec_dir = os.path.join(self.db_dir, "data")
        # self.ann_dir = os.path.join(self.db_dir, "ref")
        # self.data_dir = self.rec_dir
        # self.ref_dir = self.ann_dir

        # self.subgroups = ED({
        #     "N":  ["A01", "A03", "A05", "A06",],
        #     "V":  ["A02", "A08"],
        #     "S":  ["A09", "A10"],
        #     "VS": ["A04", "A07"],
        # })

        # self.palette = {"spb": "yellow", "pvc": "red",}

