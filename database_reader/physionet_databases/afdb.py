# -*- coding: utf-8 -*-
"""
"""
import os
import json
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, Dict, Sequence, NoReturn
from numbers import Real

import numpy as np
import pandas as pd
import wfdb
from easydict import EasyDict as ED

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "AFDB",
]


class AFDB(PhysioNetDataBase):
    """ partly finished, checked, to improve,

    MIT-BIH Atrial Fibrillation Database

    ABOUT afdb:
    -----------
    1. contains 25 long-term (each 10 hours) ECG recordings of human subjects with atrial fibrillation (mostly paroxysmal)
    2. 23 records out of 25 include the two ECG signals, the left 2 records 00735 and 03665 are represented only by the rhythm (.atr) and unaudited beat (.qrs) annotation files
    3. signals are sampled at 250 samples per second with 12-bit resolution over a range of ±10 millivolts, with a typical recording bandwidth of approximately 0.1 Hz to 40 Hz
    4. 4 classes of rhythms are annotated:
        - AFIB:  atrial fibrillation
        - AFL:   atrial flutter
        - J:     AV junctional rhythm
        - N:     all other rhythms

    NOTE:
    -----
    1. beat annotation files (.qrs files) were prepared using an automated detector and have NOT been corrected manually
    2. for some records, manually corrected beat annotation files (.qrsc files) are available

    ISSUES:
    -------

    Usage:
    ------
    1. AF detection

    References:
    -----------
    [1] https://physionet.org/content/afdb/1.0.0/
    [2] Moody GB, Mark RG. A new method for detecting atrial fibrillation using R-R intervals. Computers in Cardiology. 10:227-230 (1983).
    """
    def __init__(self, db_dir:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """ finished, checked,

        Parameters:
        -----------
        db_dir: str, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='afdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 250
        self.data_ext = "dat"
        self.ann_ext = "atr"
        self.auto_beat_ann_ext = "qrs"
        self.manual_beat_ann_ext = "qrsc"

        self.all_leads = ["ECG1", "ECG2"]

        self._ls_rec()
        self.special_records = ["00735", "03665"]
        self.qrsc_records = get_record_list_recursive(self.db_dir, self.manual_beat_ann_ext)

        self.class_map = ED(
            AFIB=1, AFL=2, J=3, N=0  # an extra isoelectric
        )


    def get_subject_id(self, rec:str) -> int:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        sid: int,
            the `get_subject_id` corr. to `rec`
        """
        raise NotImplementedError


    @property
    def all_records(self) -> List[str]:
        """ finished, checked,
        """
        if self._all_records is None:
            self._ls_rec()
        return self._all_records


    def load_data(self, rec:str, leads:Optional[Union[str, List[str]]]=None, data_format:str='channel_first', units:str='mV', freq:Optional[Real]=None) -> np.ndarray:
        """ finished, checked,

        load physical (converted from digital) ecg data,
        which is more understandable for humans

        Parameters:
        -----------
        rec: str,
            name of the record
        leads: str or list of str, optional,
            the leads to load
        data_format: str, default 'channel_first',
            format of the ecg data,
            'channel_last' (alias 'lead_last'), or
            'channel_first' (alias 'lead_first')
        units: str, default 'mV',
            units of the output signal, can also be 'μV', with an alias of 'uV'
        freq: real number, optional,
            if not None, the loaded data will be resampled to this frequency
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        fp = os.path.join(self.db_dir, rec)
        if not leads:
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads
        # p_signal in the format of 'lead_last'
        data = wfdb.rdrecord(fp, physical=True, channel_names=_leads).p_signal
        if units.lower() in ['μV', 'uV']:
            data = 1000 * data
        if freq is not None and freq != self.freq:
            data = resample_poly(data, freq, self.freq, axis=0)
        if data_format.lower() in ['channel_first', 'lead_first']:
            data = data.T
        return data

    
    def load_ann(self, rec:str, fmt:str="interval") -> Union[Dict[str, list], np.ndarray]:
        """ finished, checked,

        load annotations (header) stored in the .hea files
        
        Parameters:
        -----------
        rec: str,
            name of the record
        fmt: str, default "interval", case insensitive,
            format of returned annotation, can also be "mask"
        
        Returns:
        --------
        ann, dict or ndarray,
            the annotations in the format of intervals, or in the format of mask
        """
        fp = os.path.join(self.db_dir, rec)
        wfdb_ann = wfdb.rdann(fp, extension=self.ann_ext)
        header = wfdb.rdheader(fp)
        ann = ED({k:[] for k in self.class_map.keys()})
        critical_points = wfdb_ann.sample.tolist() + [header.sig_len]
        for idx, rhythm in enumerate(wfdb_ann.aux_note):
            ann[rhythm.replace("(", "")].append([critical_points[idx], critical_points[idx+1]])
        if fmt.lower() == "mask":
            tmp = ann.copy()
            ann = np.full(shape=(header.sig_len,), fill_value=self.class_map.N, dtype=int)
            for rhythm, l_itv in tmp.items():
                for itv in l_itv:
                    ann[itv[0]: itv[1]] = self.class_map[rhythm]
        return ann


    def load_beat_ann(self, rec:str, use_manual:bool=True) -> np.ndarray:
        """ finished, checked,

        load annotations (header) stored in the .hea files
        
        Parameters:
        -----------
        rec: str,
            name of the record
        use_manual: bool, default True,
            use manually annotated beat annotations (qrs),
            instead of those generated by algorithms
        
        Returns:
        --------
        ann, ndarray,
            locations (indices) of the qrs complexes
        """
        fp = os.path.join(self.db_dir, rec)
        if use_manual and rec in self.qrsc_records:
            ann = wfdb.rdann(fp, extension=self.manual_beat_ann_ext)
        else:
            ann = wfdb.rdann(fp, extension=self.auto_beat_ann_ext)
        ann = ann.sample
        return ann


    def plot(self, rec:str, data:Optional[np.ndarray]=None, ticks_granularity:int=0, leads:Optional[Union[str, List[str]]]=None, same_range:bool=False, waves:Optional[Dict[str, Sequence[int]]]=None, **kwargs) -> NoReturn:
        """ NOT finished,

        plot the signals of a record or external signals (units in μV),
        with metadata (freq, labels, tranche, etc.),
        possibly also along with wave delineations

        Parameters:
        -----------
        rec: str,
            name of the record
        data: ndarray, optional,
            (12-lead) ecg signal to plot,
            should be of the format "channel_first", and compatible with `leads`
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads: str or list of str, optional,
            the leads to plot
        same_range: bool, default False,
            if True, forces all leads to have the same y range
        waves: dict, optional,
            indices of the wave critical points, including
            'p_onsets', 'p_peaks', 'p_offsets',
            'q_onsets', 'q_peaks', 'r_peaks', 's_peaks', 's_offsets',
            't_onsets', 't_peaks', 't_offsets'
        kwargs: dict,

        TODO:
        -----
        1. slice too long records, and plot separately for each segment
        2. plot waves using `axvspan`

        NOTE:
        -----
        `Locator` of `plt` has default `MAXTICKS` equal to 1000,
        if not modifying this number, at most 40 seconds of signal could be plotted once

        Contributors: Jeethan, and WEN Hao
        """
        raise NotImplementedError


    def _auto_infer_units(self, data:np.ndarray) -> str:
        """ finished, checked,

        automatically infer the units of `data`,
        under the assumption that `data` not raw data, with baseline removed

        Parameters:
        -----------
        data: ndarray,
            the data to infer its units

        Returns:
        --------
        units: str,
            units of `data`, 'μV' or 'mV'
        """
        _MAX_mV = 20  # 20mV, seldom an ECG device has range larger than this value
        max_val = np.max(np.abs(data))
        if max_val > _MAX_mV:
            units = 'μV'
        else:
            units = 'mV'
        return units
