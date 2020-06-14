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
    "SLPDB",
]


class SLPDB(PhysioNetDataBase):
    """ Finished, to be improved,

    MIT-BIH Polysomnographic Database

    ABOUT slpdb:
    ------------
    1. slpdb contains over 80 hours' worth of four-, six-, and seven-channel polysomnographic (PSG) records
    2. each record has an ECG signal annotated beat-by-beat, and EEG and respiration signals annotated w.r.t. sleep stages and apnea
    3. all 16 subjects were male, aged 32 to 56 (mean age 43), with weights ranging from 89 to 152 kg (mean weight 119 kg)
    4. Records 'slp01a' and 'slp01b' are segments of one subject's polysomnogram, separated by a gap of about one hour; records 'slp02a' and 'slp02b' are segments of another subject's polysomnogram, separated by a ten-minute gap
    5. Data files have an attribute 'comments' which contains age, gender, and weight (in kg) of the subjects
    6. calibration originally provided for the BP signal of record slp37 is incorrect (since it yielded negative BPs). slp37.hea now contains an estimated BP calibration that yields more plausible BPs; these should not be regarded as accurate

    NOTE:
    -----

    ISSUES:
    -------
    1. it is weird that record 'slp45' has annotations 'M\x00' which is not in the table, should be 'MT'?

    Usage:
    ------
    1. sleep stage
    2. sleep apnea

    References:
    -----------
    [1] https://physionet.org/content/slpdb/1.0.0/
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
        super().__init__(db_name='slpdb', db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 250  # for ecg
        try:
            self.all_records = wfdb.get_record_list('slpdb')
        except:
            try:
                self.all_records = os.listdir(self.db_path)
                self.all_records = list(set([os.path.splitext(item)[0] for item in self.all_records]))
            except:
                self.all_records = ['slp01a', 'slp01b', 'slp02a', 'slp02b', 'slp03', 'slp04', 'slp14', 'slp16', 'slp32', 'slp37', 'slp41', 'slp45', 'slp48', 'slp59', 'slp60', 'slp61', 'slp66', 'slp67x']
        self.epoch_len_t = 30  # 30 seconds
        self.epoch_len = self.epoch_len_t * self.freq
        self.all_ann_states = ['1', '2', '3', '4', 'M', 'MT', 'R', 'W']
        """
        W	--- awake
        1	--- sleep stage 1
        2	--- sleep stage 2
        3	--- sleep stage 3
        4	--- sleep stage 4
        R	--- REM sleep
        MT	--- movement time

        it is weird that record 'slp45' has annotations 'M\x00' which is not in the table, should be 'MT'?
        """
        self.sleep_stage_protocol = kwargs.get('sleep_stage_protocol', 'new')
        if self.sleep_stage_protocol == "old":
            self.stage_names = ['W', 'R', 'N1', 'N2', 'N3', 'N4']
        elif self.sleep_stage_protocol == "new":
            self.stage_names = ['W', 'R', 'N1', 'N2', 'N3']
        elif self.sleep_stage_protocol == "simplified":
            self.stage_names = ['W', 'R', 'N1', 'N2']
        else:
            raise ValueError("No stage protocol named {}".format(self.sleep_stage_protocol))
        self._to_simplified_states = {'W':0, 'MT':0, 'M':0, 'R':1, '1':2, '2':2, '3':3, '4':3}
        """
        0   --- awake
        1   --- REM
        2   --- NREM1/2, shallow sleep
        3   --- NREM3/4, deep sleep
        """
        self._to_new_aasm_states = {'W':0, 'MT':0, 'M':0, 'R':1, '1':2, '2':3, '3':4, '4':4}
        """
        0   --- awake
        1   --- REM
        2   --- N1
        3   --- N2
        4   --- N3
        """
        self._to_old_aasm_states = {'W':0, 'MT':0, 'M':0, 'R':1, '1':2, '2':3, '3':4, '4':5}
        """
        0   --- awake
        1   --- REM
        2   --- N1
        3   --- N2
        4   --- N3
        5   --- N4
        """

    
    def get_subject_id(self, rec) -> int:
        """

        """
        _rn = rec+'n'
        head = '31000'
        to_tail = lambda t: t[3:5]+{'n':'0','an':'1','bn':'2','xn':'9'}[t[5:]]
        pid = int(head+to_tail(_rn))
        return pid


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
