# -*- coding: utf-8 -*-
"""
"""
import os
import json
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, NoReturn
from numbers import Real

import numpy as np
import pandas as pd
import wfdb

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "AFDB",
]


class AFDB(PhysioNetDataBase):
    """ NOT Finished,

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
    1. 

    References:
    -----------
    [1] https://physionet.org/content/afdb/1.0.0/
    [2] Moody GB, Mark RG. A new method for detecting atrial fibrillation using R-R intervals. Computers in Cardiology. 10:227-230 (1983).
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
        super().__init__(db_name='afdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 250
        self.data_ext = "dat"
        self.ann_ext = "atr"
        self.auto_beat_ann_ext = "qrs"
        self.manual_beat_ann_ext = "qrsc"

        self._ls_rec()


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
    def all_records(self):
        """ finished, checked
        """
        if self._all_records is None:
            self._ls_rec()
        return self._all_records


    def _ls_diagnoses_records(self) -> NoReturn:
        """ finished, checked,

        list all the records for all diagnoses
        """
        raise NotImplementedError


    @property
    def diagnoses_records_list(self):
        """ finished, checked
        """
        if self._diagnoses_records_list is None:
            self._ls_diagnoses_records()
        return self._diagnoses_records_list


    def load_data(self, rec:str, leads:Optional[Union[str, List[str]]]=None, data_format:str='channel_first', backend:str='wfdb', units:str='mV', freq:Optional[Real]=None) -> np.ndarray:
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
        backend: str, default 'wfdb',
            the backend data reader, can also be 'scipy'
        units: str, default 'mV',
            units of the output signal, can also be 'μV', with an alias of 'uV'
        freq: real number, optional,
            if not None, the loaded data will be resampled to this frequency
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        raise NotImplementedError

    
    def load_ann(self, rec:str, raw:bool=False, backend:str="wfdb") -> Union[dict,str]:
        """ finished, checked,

        load annotations (header) stored in the .hea files
        
        Parameters:
        -----------
        rec: str,
            name of the record
        raw: bool, default False,
            if True, the raw annotations without parsing will be returned
        backend: str, default "wfdb", case insensitive,
            if is "wfdb", `wfdb.rdheader` will be used to load the annotations;
            if is "naive", annotations will be parsed from the lines read from the header files
        
        Returns:
        --------
        ann_dict, dict or str,
            the annotations with items: ref. `self.ann_items`
        """
        raise NotImplementedError


    def load_header(self, rec:str, raw:bool=False) -> Union[dict,str]:
        """
        alias for `load_ann`, as annotations are also stored in header files
        """
        raise NotImplementedError


    def plot(self, rec:str, data:Optional[np.ndarray]=None, ticks_granularity:int=0, leads:Optional[Union[str, List[str]]]=None, same_range:bool=False, waves:Optional[Dict[str, Sequence[int]]]=None, **kwargs) -> NoReturn:
        """ finished, checked, to improve,

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
            the granularity to plot axis ticks, the higher the more
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


    @staticmethod
    def get_arrhythmia_knowledge(arrhythmias:Union[str,List[str]], **kwargs) -> NoReturn:
        """ finished, checked,

        knowledge about ECG features of specific arrhythmias,

        Parameters:
        -----------
        arrhythmias: str, or list of str,
            the arrhythmia(s) to check, in abbreviations or in SNOMED CT Code
        """
        if isinstance(arrhythmias, str):
            d = [normalize_class(arrhythmias)]
        else:
            d = [normalize_class(c) for c in arrhythmias]
        # pp = pprint.PrettyPrinter(indent=4)
        # unsupported = [item for item in d if item not in dx_mapping_all['Abbreviation']]
        unsupported = [item for item in d if item not in dx_mapping_scored['Abbreviation'].values]
        assert len(unsupported) == 0, \
            f"`{unsupported}` {'is' if len(unsupported)==1 else 'are'} not supported!"
        for idx, item in enumerate(d):
            # pp.pprint(eval(f"EAK.{item}"))
            print(dict_to_str(eval(f"EAK.{item}")))
            if idx < len(d)-1:
                print("*"*110)

