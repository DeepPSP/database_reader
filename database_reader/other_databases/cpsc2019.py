# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, NoReturn
from numbers import Real

from database_reader.utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from database_reader.base import OtherDataBase


__all__ = [
    "CPSC2019",
]


class CPSC2019(OtherDataBase):
    """

    The 2nd China Physiological Signal Challenge (CPSC 2019):
    Challenging QRS Detection and Heart Rate Estimation from Single-Lead ECG Recordings

    ABOUT CPSC2019:
    ---------------
    1. Training data consists of 2,000 single-lead ECG recordings collected from patients with cardiovascular disease (CVD)
    2. Each of the recording last for 10 s
    3. Sampling rate = 500 Hz

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. ecg wave delineation

    References:
    -----------
    [1] http://2019.icbeb.org/Challenge.html
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """ finished, to be improved,

        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="CPSC2019", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        
        self.freq = 500
        self.spacing = 1000 / self.freq

        self.all_records = []


    def get_patient_id(self, rec_no:int) -> int:
        """ not finished,

        Parameters:
        -----------
        rec_no: int,
            number of the record, NOTE that rec_no starts from 1

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

    
    def load_data(self, rec_no:int, keep_dim:bool=True) -> np.ndarray:
        """ not finished, not checked,

        Parameters:
        -----------
        rec_no: int,
            number of the record, NOTE that rec_no starts from 1
        keep_dim: bool, default True,
            whether or not to flatten the data of shape (n,1)
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        raise NotImplementedError


    def load_ann(self, rec_no:int) -> Dict[str, np.ndarray]:
        """ not finished, not checked,

        Parameters:
        -----------
        rec_no: int,
            number of the record, NOTE that rec_no starts from 1
        
        Returns:
        --------
        ann: dict,
            with items "SPB_indices" and "PVC_indices", which record the indices of SPBs and PVCs
        """
        raise NotImplementedError


    def plot(self, rec_no:int, **kwargs) -> NoReturn:
        """ not finished, not checked,

        Parameters:
        -----------
        rec_no: int,
            number of the record, NOTE that rec_no starts from 1
        """
        raise NotImplementedError
