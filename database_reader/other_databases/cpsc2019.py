# -*- coding: utf-8 -*-
"""
"""
import os
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, NoReturn
from numbers import Real

import numpy as np
import pandas as pd
from scipy.io import loadmat

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
    1. there're 13 records with unusual large values (> 20 mV):
        data_00098, data_00167, data_00173, data_00223, data_00224, data_00245, data_00813,
        data_00814, data_00815, data_00833, data_00841, data_00949, data_00950
    >>> for rec in dr.all_records:
    >>>     data = dr.load_data(rec)
    >>>     if np.max(data) > 20:
    >>>         print(f"{rec} has max value ({np.max(data)} mV) > 20 mV")
    ... data_00173 has max value (32.72031811111111 mV) > 20 mV
    ... data_00223 has max value (32.75516713333333 mV) > 20 mV
    ... data_00224 has max value (32.7519272 mV) > 20 mV
    ... data_00245 has max value (32.75305293939394 mV) > 20 mV
    ... data_00813 has max value (32.75865595876289 mV) > 20 mV
    ... data_00814 has max value (32.75865595876289 mV) > 20 mV
    ... data_00815 has max value (32.75558282474227 mV) > 20 mV
    ... data_00833 has max value (32.76330123809524 mV) > 20 mV
    ... data_00841 has max value (32.727626558139534 mV) > 20 mV
    ... data_00949 has max value (32.75699667692308 mV) > 20 mV
    ... data_00950 has max value (32.769551661538465 mV) > 20 mV
    2. rpeak references (annotations) loaded from files has dtype = uint16,
    which would produce unexpected large positive values when subtracting values larger than it,
    rather than the correct negative value.
    This might cause confusion in computing metrics when using annotations subtracting
    (instead of being subtracted by) predictions.

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
        self._all_records = [f"data_{i:05d}" for i in range(1,1+self.nb_records)]
        self._all_annotations = [f"R_{i:05d}" for i in range(1,1+self.nb_records)]
        # self.all_references = self.all_annotations
        self.rec_dir = os.path.join(self.db_dir, "data")
        self.ann_dir = os.path.join(self.db_dir, "ref")
        self.data_dir = self.rec_dir
        self.ref_dir = self.ann_dir


    @property
    def all_records(self):
        """
        """
        return self._all_records

    @property
    def all_annotations(self):
        """
        """
        return self._all_annotations

    @property
    def all_references(self):
        """
        """
        return self._all_annotations


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
        """ finished, checked,

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
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        keep_dim: bool, default True,
            whether or not to flatten the data of shape (n,1)
        
        Returns:
        --------
        ann: dict,
            with items "SPB_indices" and "PVC_indices", which record the indices of SPBs and PVCs
        """
        fp = os.path.join(self.ann_dir, f"{self._get_ann_name(rec)}{self.ann_ext}")
        ann = loadmat(fp)["R_peak"].astype(int)
        if not keep_dim:
            ann = ann.flatten()
        return ann


    def load_rpeaks(self, rec:Union[int,str], keep_dim:bool=True) -> Dict[str, np.ndarray]:
        """
        alias of `self.load_ann`
        """
        return self.load_ann(rec=rec, keep_dim=keep_dim)


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


    def _get_ann_name(self, rec:Union[int,str]) -> str:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name

        Returns:
        --------
        ann_name: str,
            filename of annotations of the record `rec`
        """
        rec_name = self._get_rec_name(rec)
        ann_name = rec_name.replace("data", "R")
        return ann_name


    def plot(self, rec:Union[int,str], data:Optional[np.ndarray]=None, ticks_granularity:int=0) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        data: ndarray, optional,
            ecg signal to plot,
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        """
        if 'plt' not in dir():
            import matplotlib.pyplot as plt

        if data is None:
            _data = self.load_data(rec, units="μV", keep_dim=False)
        else:
            units = self._auto_infer_units(data)
            if units == "mV":
                _data = data * 1000
            elif units == "μV":
                _data = data.copy()

        duration = len(_data) / self.freq
        secs = np.linspace(0, duration, len(_data))
        rpeak_secs = self.load_rpeaks(rec, keep_dim=False) / self.freq

        fig_sz_w = int(DEFAULT_FIG_SIZE_PER_SEC * duration)
        y_range = np.max(np.abs(_data))
        fig_sz_h = 6 * y_range / 1500
        fig, ax = plt.subplots(figsize=(fig_sz_w, fig_sz_h))
        ax.plot(secs, ——data, c='black')
        ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
        if ticks_granularity >= 1:
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax.yaxis.set_major_locator(plt.MultipleLocator(500))
            ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        if ticks_granularity >= 2:
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(100))
            ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        for r in rpeak_secs:
            ax.axvspan(r-0.01, r+0.01, color='green', alpha=0.7)
        ax.set_xlim(secs[0], secs[-1])
        ax.set_ylim(-y_range, y_range)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [μV]')
        plt.show()
