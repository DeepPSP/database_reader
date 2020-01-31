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
    "UCDDB",
]


class UCDDB(PhysioNetDataBase):
    """ NOT finished,

    St. Vincent's University Hospital / University College Dublin Sleep Apnea Database

    About ucddb:
    ------------
    1. contains 25 full overnight polysomnograms with simultaneous three-channel Holter ECG
    2. PSG signals recorded were:
        EEG (C3-A2),
        EEG (C4-A1),
        left EOG,
        right EOG,
        submental EMG,
        ECG (modified lead V2),
        oro-nasal airflow (thermistor),
        ribcage movements,
        abdomen movements (uncalibrated strain gauges),
        oxygen saturation (finger pulse oximeter),
        snoring (tracheal microphone),
        body position
    3. holter ECG leads: V5, CC5, V5R
    4. sleep stage annotations:
        0 - Wake
        1 - REM
        2 - Stage 1
        3 - Stage 2
        4 - Stage 3
        5 - Stage 4
        6 - Artifact
        7 - Indeterminate
    5. respiratory events:
        O - obstructive, apneas,
        C - central apneas,
        M - mixed apneas,
        HYP - hypopneas,
        PB - periodic breathing episodes
        CS - Cheynes-Stokes
        Bradycardia / Tachycardia

    NOTE:
    -----
    1. in record ucddb002, only two distinct ECG signals were recorded; the second ECG signal was also used as the third signal.

    ISSUES:
    -------

    Usage:
    ------
    1. 

    References:
    -----------
    [1] https://physionet.org/content/ucddb/1.0.0/
    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='ucddb', db_path=db_path, **kwargs)


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
