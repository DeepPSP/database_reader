# -*- coding: utf-8 -*-
"""
"""
import io
import os
import pprint
import wfdb
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, NoReturn
from numbers import Real

from database_reader.utils import (
    ArrayLike,
    PVC, SPB,
)
from database_reader.base import OtherDataBase


__all__ = [
    "CPSC2020",
]


class CPSC2020(OtherDataBase):
    """

    The 3rd China Physiological Signal Challenge 2020:
    Searching for Premature Ventricular Contraction (PVC) and Supraventricular Premature Beat (SPB) from Long-term ECGs

    ABOUT CPSC2019:
    ---------------
    1. training data consists of 10 single-lead ECG recordings collected from arrhythmia patients, each of the recording last for about 24 hours
    2. A02, A03, A08 are patient with atrial fibrillation
    3. sampling frequency = 400 Hz
    4. about PVC and SPB, ref utils.utils_misc.ecg_arrhythmia_knowledge
    5. Detailed information:
        rec   ?AF   Length(h)   # N beats   # V beats   # S beats   # Total beats
        A01   No	25.89       109,062     0           24          109,086
        A02   Yes	22.83       98,936      4,554       0           103,490
        A03   Yes	24.70       137,249     382         0           137,631
        A04   No	24.51       77,812      19,024      3,466       100,302
        A05   No	23.57       94,614  	1	        25	        94,640
        A06   No	24.59       77,621  	0	        6	        77,627
        A07   No	23.11	    73,325  	15,150	    3,481	    91,956
        A08   Yes	25.46	    115,518 	2,793	    0	        118,311
        A09   No	25.84	    88,229  	2	        1,462	    89,693
        A10   No	23.64	    72,821	    169	        9,071	    82,061

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. ecg arrhythmia (PVC, SPP) detection

    References:
    -----------
    [1] http://www.icbeb.org/CPSC2020.html
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """ finished, to be improved,

        Parameters:
        -----------
        db_path: str,
            storage path of the database
        verbose: int, default 2,
        """
        super().__init__(db_name="CPSC2020", db_path=db_path, verbose=verbose, **kwargs)

        self.freq = 400
        self.spacing = 1000/self.freq
        self.rec_ext = '.mat'
        self.ann_ext = '.hea'

        self.nb_records = 10
        self.all_records = ["A{0:02d}".format(i) for i in range(1,1+self.nb_records)]
        self.all_annotations = ["R{0:02d}".format(i) for i in range(1,1+self.nb_records)]
        self.all_references = self.all_annotations
        self.rec_folder = os.path.join(self.db_path, "data")
        self.ann_folder = os.path.join(self.db_path, "ref")
        self.ref_folder = self.ann_folder


    def get_patient_id(self, rec_no:int) -> int:
        """ not finished,

        Parameters:
        -----------
        rec_no: int,
            number of the record, or 'subject_ID'

        Returns:
        --------
        pid: int,
            the `patient_id` corr. to `rec_no`
        """
        pid = 0
        raise NotImplementedError
    

    def database_info(self, detailed:bool=False) -> NoReturn:
        """ not finished,

        print the information about the database

        Parameters:
        -----------
        detailed: bool, default False,
            if False, an short introduction of the database will be printed,
            if True, then docstring of the class will be printed additionally
        """
        raw_info = {}

        print(raw_info)
        
        if detailed:
            print(self.__doc__)


    def load_data(self, rec_no:int) -> np.ndarray:
        """ finished, not checked,

        Parameters:
        -----------
        rec_no: int,
            number of the record, NOTE that rec_no starts from 1
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        assert rec_no in range(1, self.nb_records+1), "rec_no should be in range(1,{})".format(self.nb_records+1)
        rec_fp = os.path.join(self.rec_folder, self.all_records[rec_no-1] + self.rec_ext)
        data = (1000 * loadmat(rec_fp).flatten()).astype(int)
        return data


    def load_ann(self, rec_no:int) -> Dict[str, np.ndarray]:
        """ finished, not checked,

        Parameters:
        -----------
        rec_no: int,
            number of the record, NOTE that rec_no starts from 1
        
        Returns:
        --------
        ann: dict,
            with items "SPB_indices" and "PVC_indices", which record the indices of SPBs and PVCs
        """
        assert rec_no in range(1, self.nb_records+1), "rec_no should be in range(1,{})".format(self.nb_records+1)
        ann_fp = os.path.join(self.ann_folder, self.all_annotations[rec_no-1] + self.ann_ext)
        ann = loadmat(ann_fp)['ref']
        ann = {
            "SPB_indices": ann['S_ref'][0,0],
            "PVC_indices": ann['V_ref'][0,0],
        }
        return ann

    
    def plot(self, rec_no:int, sampfrom:Optional[int]=None, sampto:Optional[int]=None, ectopic_beats_only:bool=False, **kwargs) -> NoReturn:
        """ not finished, not checked,

        Parameters:
        -----------
        rec_no: int,
            number of the record, NOTE that rec_no starts from 1
        """
        raise NotImplementedError
