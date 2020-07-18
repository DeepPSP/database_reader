# -*- coding: utf-8 -*-
"""
"""
import os
import time
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
    "INCARTDB",
]


class INCARTDB(PhysioNetDataBase):
    """ NOT finished,

    St Petersburg INCART 12-lead Arrhythmia Database

    ABOUT incartdb:
    ---------------
    1. consists of 75 annotated recordings extracted from 32 Holter records, each of 12 leads and of length 30 minutes.
    2. sampling frequency is 257 Hz
    3. ADC gain ranges from 250 to 1100
    4. annotations are beat-wise, totaling a number of 175,000 beats
    5. diagnoses distribution:
        Diagnosis	                                    # Patients
        Acute MI	                                    2
        Transient ischemic attack (angina pectoris)	    5
        Prior MI	                                    4
        Coronary artery disease with hypertension  	    7 (4 with left ventricular hypertrophy)
        Sinus node dysfunction	                        1
        Supraventricular ectopy	                        18
        Atrial fibrillation or SVTA	                    3 (2 with paroxysmal AF)
        WPW	                                            2
        AV block                                        1
        Bundle branch block                             3

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. ECG arrhythmia detection

    References:
    -----------
    [1] https://physionet.org/content/incartdb/1.0.0/
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
        super().__init__(db_name='incartdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 257
        self.spacing = 1000/self.freq

        self.data_ext = '.dat'
        self.ann_ext = '.atr'
        self.aux_ann_ext = '.hea'

        self._ls_rec()

        # this file links record names with patient's `subject_id`
        self.patients_file = os.path.join(self.db_dir, 'files-patients-diagnoses.txt')
        # this file decribes each record's diagnosis
        self.record_description_file = os.path.join(self.db_dir, 'files-patients-diagnoses.txt')
        

    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)


    def load_data(self, rec:str, data_format='channels_last') -> np.ndarray:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record
        data_format: str, default 'channels_last',
            format of the ecg data, 'channels_last' or 'channels_first' (original)
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        raise NotImplementedError

    def load_ann(self, rec:str) -> dict:
        """ finished, checked,
        
        Parameters:
        -----------
        rec: str,
            name of the record
        
        Returns:
        --------
        ann_dict, dict,
            the annotations with items: ref. self.ann_items
        """
        raise NotImplementedError
