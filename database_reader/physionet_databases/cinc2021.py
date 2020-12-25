# -*- coding: utf-8 -*-
"""
"""
import os, io, sys
import re
import json
import time
import logging
# import pprint
from copy import deepcopy
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Tuple, Set, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import wfdb
from scipy.io import loadmat
from scipy.signal import resample, resample_poly
from easydict import EasyDict as ED

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
    get_record_list_recursive3,
    ms2samples, samples2ms,
)
from ..utils.utils_misc import ecg_arrhythmia_knowledge as EAK
from ..utils.utils_universal.utils_str import dict_to_str
from ..base import PhysioNetDataBase


__all__ = [
    "CINC2021",
]


class CINC2021(PhysioNetDataBase):
    """ NOT finished,

    Will Two Do? Varying Dimensions in Electrocardiography:
    The PhysioNet/Computing in Cardiology Challenge 2021

    ABOUT CINC2021:
    ---------------
    0. goal: build an algorithm that can classify cardiac abnormalities from either
        - twelve-lead (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)
        - six-lead (I, II, III, aVL, aVR, and aVF),
        - two-lead (II and V5)
    ECG recordings.
    1. tranches of data:
        - CPSC2018 (tranches A and B of CINC2020):
            contains 13,256 ECGs,
            10,330 ECGs shared as training data, 1,463 retained as validation data,
            and 1,463 retained as test data.
            Each recording is between 6 and 144 seconds long with a sampling frequency of 500 Hz
        - INCARTDB (tranche C of CINC2020):
            contains 75 annotated ECGs,
            all shared as training data, extracted from 32 Holter monitor recordings.
            Each recording is 30 minutes long with a sampling frequency of 257 Hz
        - PTB (PTB and PTB-XL, tranches D and E of CINC2020):
            contains 22,353 ECGs,
            516 + 21,837, all shared as training data.
            Each recording is between 10 and 120 seconds long,
            with a sampling frequency of either 500 (PTB-XL) or 1,000 (PTB) Hz
        - Georgia (tranche F of CINC2020):
            contains 20,678 ECGs,
            10,334 ECGs shared as training data, 5,167 retained as validation data,
            and 5,167 retained as test data.
            Each recording is between 5 and 10 seconds long with a sampling frequency of 500 Hz
        - American (NEW, UNDISCLOSED):
            contains 10,000 ECGs,
            all retained as test data,
            geographically distinct from the Georgia database.
            Perhaps is the main part of the hidden test set of CINC2020
    3. to add more, or ref. docstring of `CINC2020`

    Usage:
    ------
    1. ECG arrhythmia detection

    References:
    -----------
    [1] https://physionetchallenges.github.io/2021/
    [1] https://physionetchallenges.github.io/2020/
    [2] http://2018.icbeb.org/#
    [3] https://physionet.org/content/incartdb/1.0.0/
    [4] https://physionet.org/content/ptbdb/1.0.0/
    [5] https://physionet.org/content/ptb-xl/1.0.1/
    [6] https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            print and log verbosity
        """
        super().__init__(db_name="CINC2021", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        
        self.rec_ext = "mat"
        self.ann_ext = "hea"

    def load_data(self, ):
        """
        """
        raise NotImplementedError

    def load_ann(self, ):
        """
        """
        raise NotImplementedError

    def plot(self, ):
        """
        """
        raise NotImplementedError
