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

from easydict import EasyDict as ED

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "MIMIC3",
]


class MIMIC3WDB(PhysioNetDataBase):
    """ NOT Finished,

    MIMIC-III Waveform Database

    ABOUT mimic3wdb:
    ----------------
    1. contains 67,830 record sets for approximately 30,000 ICU patients
    2. almost all record sets include a waveform record containing digitized signals (typically including ECG, ABP, respiration, and PPG, and frequently other signals) and a “numerics” record containing time series (HR, RESP, SpO2, BP, etc.) of periodic measurements
    3. a subset (the matched subset) of mimic3wdb contains waveform and numerics records that have been matched and time-aligned with MIMIC-III Clinical Database records

    NOTE:
    -----

    ISSUES:
    -------
    ref. [3]

    Usage:
    ------
    1. 

    References:
    -----------
    [1] https://mimic.physionet.org/
    [2] https://github.com/MIT-LCP/mimic-code
    [3] https://www.physionet.org/content/mimiciii/1.4/
    [4] https://archive.physionet.org/physiobank/database/mimic3wdb/
    [5] https://archive.physionet.org/physiobank/database/mimic3wdb/matched/
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
            print verbosity
        kwargs:
        """
        super().__init__(db_name='mimic3wdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.metadata_files = ED(
            all_records=os.path.join(self.db_dir, "RECORDS"),
            waveforms=os.path.join(self.db_dir, "RECORDS-waveforms"),
            numerics=os.path.join(self.db_dir, "RECORDS-numerics"),
            adults=os.path.join(self.db_dir, "RECORDS-adults"),
            neonates=os.path.join(self.db_dir, "RECORDS-neonates"),
        )
        self.freq = None   # typically 125
        
        self.data_ext = "dat"
        self.ann_ext = "hea"
        self._ls_rec()


    def load_data(self,):
        """
        """
        raise NotImplementedError


    def _ls_rec(self) -> NoReturn:
        """
        """
        try:
            tmp = wfdb.get_record_list(self.db_name)
        except:
            with open(self.metadata_files["all_records"], "r") as f:
                tmp = f.read().splitlines()
        self._all_records = {}
        for l in tmp:
            gp, sb = l.strip("/").split("/")
            if gp in self._all_records.keys():
                self._all_records[gp].append(sb)
            else:
                self._all_records[gp] = [sb]
        self._all_records = {k:sorted(v) for k,v in self._all_records.items()}


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError
