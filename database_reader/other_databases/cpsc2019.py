# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, NoReturn
from numbers import Real

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
    DEFAULT_FIG_SIZE_PER_SEC,
)
from ..base import OtherDataBase


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

        self.rec_ext = '.mat'
        self.ann_ext = '.mat'

        self.nb_records = 2000
        self.all_records = [f"data_{i:05d}" for i in range(1,1+self.nb_records)]
        self.all_annotations = [f"R_{i:05d}" for i in range(1,1+self.nb_records)]
        self.all_references = self.all_annotations
        self.rec_dir = os.path.join(self.db_dir, "data")
        self.ann_dir = os.path.join(self.db_dir, "ref")
        self.data_dir = self.rec_dir
        self.ref_dir = self.ann_dir


    def get_subject_id(self, rec_no:int) -> int:
        """ not finished,

        Parameters:
        -----------
        rec_no: int,
            number of the record, NOTE that rec_no starts from 1

        Returns:
        --------
        pid: int,
            the `subject_id` corr. to `rec_no`
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

    
    def load_data(self, rec:Union[int,str], units:str='mV', keep_dim:bool=True) -> np.ndarray:
        """ finished, not checked,

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
        fp = os.path.join(self.data_dir, f"{self._get_rec_name(rec)}{self.rec_ext}")
        data = loadmat(fp)["ecg"]
        if units.lower() in ["uv", "μv",]:
            data = (1000 * data).astype(int)
        if not keep_dim:
            data = data.flatten()
        return data


    def load_ann(self, rec:Union[int,str], keep_dim:bool=True) -> Dict[str, np.ndarray]:
        """ finished, not checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        
        Returns:
        --------
        ann: dict,
            with items "SPB_indices" and "PVC_indices", which record the indices of SPBs and PVCs
        """
        fp = os.path.join(self.data_dir, f"{self._get_rec_name(rec)}{self.ann_ext}")
        ann = loadmat(fp)["R_peak"]
        if not keep_dim:
            ann = ann.flatten()
        return ann


    def _get_rec_name(self, rec:Union[int,str]) -> str:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name

        Returns:
        --------
        rec_name: str,
            filename of the record
        """
        if isinstance(rec, int):
            assert rec in range(1, self.nb_records+1), f"rec should be in range(1,{self.nb_records+1})"
            rec_name = self.all_records[rec-1]
        elif isinstance(rec, str):
            assert rec in self.all_records, f"rec {rec} not found"
            rec_name = rec
        return rec_name


    def plot(self, rec:Union[int,str], ticks_granularity:int=0) -> NoReturn:
        """ finished, not checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        ticks_granularity: int, default 0,
            granularity of ticks,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        """
        if 'plt' not in dir():
            import matplotlib.pyplot as plt

        data = self.load_data(rec, units='uv', keep_dim=False)
        duration = len(data) / self.fs
        ann = self.load_ann(rec)
        secs = np.linspace(0, duration, len(data))

        fig_sz_w = int(DEFAULT_FIG_SIZE_PER_SEC * duration)
        y_range = np.max(np.abs(data))
        fig_sz_h = 6 * y_range / 1500
        fig, ax = plt.subplots(figsize=(fig_sz_w, fig_sz_h))
        ax.plot(secs, data, c='black')
        ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
        if ticks_granularity >= 1:
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax.yaxis.set_major_locator(plt.MultipleLocator(500))
            ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        if ticks_granularity >= 2:
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(100))
            ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set_xlim(secs[0], secs[-1])
        ax.set_ylim(-y_range, y_range)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [μV]')
        plt.show()
