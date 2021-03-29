# -*- coding: utf-8 -*-
"""
"""
import os
import random
import math
import time
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, Dict, Sequence, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
from scipy.signal import resample, resample_poly
from easydict import EasyDict as ED

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
    get_record_list_recursive3,
    ms2samples, samples2ms,
    DEFAULT_FIG_SIZE_PER_SEC,
)
from ..base import OtherDataBase


__all__ = [
    "CPSC2021",
]


class CPSC2021(OtherDataBase):
    r"""

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
    7. classification of a record is stored in corresponding .hea file, which can be accessed via the attribute `comments` of a wfdb Record obtained using `wfdb.rdheader`, `wfdb.rdrecord`, and `wfdb.rdsamp`; beat annotations and rhythm annotations can be accessed using the attributes `symbol`, `aux_note` of a wfdb Annotation obtained using `wfdb.rdann`, corresponding indices in the signal can be accessed via the attribute `sample`
    8. challenge task:
        (1). clasification of rhythm types: non-AF rhythm (N), persistent AF rhythm (AFf) and paroxysmal AF rhythm (AFp)
        (2). locating of the onset and offset for any AF episode prediction
    9. challenge metrics:
        (1) metrics (Ur, scoring matrix) for classification:
                Prediction
                N        AFf        AFp
        N      +1        -1         -0.5
        AFf    -2        +1          0
        AFp    -1         0         +1
        (2) metric (Ue) for detecting onsets and offsets for AF events (episodes):
        +1 if the detected onset (or offset) is within ±1 beat of the annotated position, and +0.5 if within ±2 beats
        (3) final score (U):
        U = \dfrac{1}{N} \sum\limits_{i=1}^N \left( Ur_i + \dfrac{Ma_i}{\max\{Mr_i, Ma_i\}} \right)
        where N is the number of records, Ma is the number of annotated AF episodes, Mr the number of predicted AF episodes

    NOTE:
    -----
    1. if an ECG record is classified as AFf, the provided onset and offset locations should be the first and last record points. If an ECG record is classified as N, the answer should be an empty list
    2. it can be inferred from the classification scoring matrix that the punishment of false negatives of AFf is very heavy, while mixing-up of AFf and AFp is not punished

    ISSUES:
    -------
    1. 

    TODO:
    -----
    1. 

    Usage:
    ------
    1. AF (event, fine) detection

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
        self.rec_ext = "dat"
        self.ann_ext = "hea"
        self.all_leads = ["I", "II"]

        self._ls_rec()

        # self.nb_records = 10

        # self.palette = {"spb": "yellow", "pvc": "red",}


    def _ls_rec(self) -> NoReturn:
        """ finished, checked,

        list all the records and load into `self._all_records`,
        facilitating further uses
        """
        fn = "RECORDS"
        record_list_fp = os.path.join(self.db_dir, fn)
        if os.path.isfile(record_list_fp):
            with open(record_list_fp, "r") as f:
                self._all_records = f.read().splitlines()
        else:
            print("Please wait patiently to let the reader find all records...")
            start = time.time()
            rec_patterns_with_ext = f"data_(?:\d+)_(?:\d+).{self.rec_ext}"
            self._all_records = \
                get_record_list_recursive3(self.db_dir, rec_patterns_with_ext)
            print(f"Done in {time.time() - start:.5f} seconds!")
            with open(record_list_fp, "w") as f:
                f.write("\n".join(self._all_records))


    def get_subject_id(self, rec:str) -> int:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        sid: int,
            subject id corresponding to the record
        """
        sid = int(rec.split("_")[1])
        return sid


    def load_data(self, rec:str, leads:Optional[Union[str, List[str]]]=None, data_format:str="channel_first", units:str="mV", fs:Optional[Real]=None) -> np.ndarray:
        """ NOT finished, NOT checked,

        load physical (converted from digital) ecg data,
        which is more understandable for humans

        Parameters:
        -----------
        rec: str,
            name of the record
        leads: str or list of str, optional,
            the leads to load
        data_format: str, default "channel_first",
            format of the ecg data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        fs: real number, optional,
            if not None, the loaded data will be resampled to this frequency
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        assert data_format.lower() in ["channel_first", "lead_first", "channel_last", "lead_last"]
        if not leads:
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads

        rec_fp = os.path.join(self.db_dir, rec)
        wfdb_rec = wfdb.rdrecord(rec_fp, physical=True, channel_names=_leads)
        data = np.asarray(wfdb_rec.p_signal.T)
        # lead_units = np.vectorize(lambda s: s.lower())(wfdb_rec.units)

        if units.lower() in ["uv", "μv"]:
            data = data * 1000

        if fs is not None and fs != self.fs:
            data = resample_poly(data, fs, self.fs, axis=1)

        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T

        return data

    
    def load_ann(self,):
        """
        """
        raise NotImplementedError


    def load_rpeaks(self,) -> np.ndarray:
        """
        """
        raise NotImplementedError


    def load_af_episodes(self,):
        """
        """
        raise NotImplementedError


    def plot(self) -> NoReturn:
        """
        """
        raise NotImplementedError
