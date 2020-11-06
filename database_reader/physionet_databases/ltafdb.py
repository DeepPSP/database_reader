# -*- coding: utf-8 -*-
"""
"""
import os
import json
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, Dict, Sequence, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import wfdb
from easydict import EasyDict as ED

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "LTAFDB",
]


class LTAFDB(PhysioNetDataBase):
    """ NOT Finished,

    Long Term AF Database

    ABOUT ltafdb:
    -------------
    1. contains 84 long-term ECG recordings of subjects with paroxysmal or sustained atrial fibrillation
    2. each record contains two simultaneously recorded ECG signals digitized at 128 Hz
    3. records have duration 24 - 25 hours
    4. qrs annotations (.qrs files) were produced by an automated QRS detector, in which detected beats (including occasional ventricular ectopic beats) are labelled 'N', detected artifacts are labelled '|', and AF terminations are labelled 'T' (inserted manually)
    5. atr annotations (.atr files) were obtained by manual review of the output of an automated ECG analysis system; in these annotation files, all detected beats are labelled by type, and rhythm changes are also annotated

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. 

    References:
    -----------
    [1] https://physionet.org/content/ltafdb/1.0.0/
    [2] Petrutiu S, Sahakian AV, Swiryn S. Abrupt changes in fibrillatory wave characteristics at the termination of paroxysmal atrial fibrillation in humans. Europace 9:466-470 (2007).
    [3] https://physionet.org/files/ltafdb/1.0.0/tables.shtml
    """
    def __init__(self, db_dir:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """ NOT finished,

        Parameters:
        -----------
        db_dir: str, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='ltafdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 128
        self.data_ext = "dat"
        self.auto_ann_ext = "qrs"
        self.manual_ann_ext = "atr"

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


    def load_ann(self, rec:str):
        """ NOT finished,

        load annotations (header) stored in the .hea files
        
        Parameters:
        -----------
        rec: str,
            name of the record
        
        Returns:
        --------
        ann,
        """
        # fp = os.path.join(self.db_dir, rec)
        # wfdb_ann = wfdb.rdann(fp, extension=self.ann_ext)
        # header = wfdb.rdheader(fp)
        # ann = ED({k:[] for k in self.class_map.keys()})
        # critical_points = wfdb_ann.sample.tolist() + [header.sig_len]
        # for idx, rhythm in enumerate(wfdb_ann.aux_note):
        #     ann[rhythm.replace("(", "")].append([critical_points[idx], critical_points[idx+1]])
        # if fmt.lower() == "mask":
        #     tmp = ann.copy()
        #     ann = np.full(shape=(header.sig_len,), fill_value=self.class_map.N, dtype=int)
        #     for rhythm, l_itv in tmp.items():
        #         for itv in l_itv:
        #             ann[itv[0]: itv[1]] = self.class_map[rhythm]
        # return ann
        raise NotImplementedError
