# -*- coding: utf-8 -*-
"""
"""
import os
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

import wfdb
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
from easydict import EasyDict as ED

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "MIMIC3WDB_MATCHED",
]


class MIMIC3WDB_MATCHED(PhysioNetDataBase):
    """ NOT Finished,

    MIMIC-III Waveform Database Matched Subset

    ABOUT mimic3wdb_matched:
    ------------------------
    1. contains 22,317 waveform records, and 22,247 numerics records, for 10,282 distinct ICU patients, which have been matched and time-aligned with MIMIC-III Clinical Database records, with total size 2.4 TB
    2. almost all record sets include waveform records (usually multi-record consisting of multiple continuous segments) from several visits (each visit is a single "folder" in "mimic3wdb") containing digitized signals (typically including ECG, ABP, respiration, and PPG, and frequently other signals) digitized at 125 Hz with 8-, 10-, or (occasionally) 12-bit resolution and "numerics" records containing time series of vital signs (HR, RESP, SpO2, BP, etc.) of periodic measurements sampled once per second or once per minute
    3. all data associated with a particular patient have been placed into a single subdirectory "matched/pXX/pXXNNNN/" where "N", "X" are digits, named according to the patient's MIMIC-III subject_ID. These subdirectories are further divided into ten intermediate-level directories (matched/p00 to matched/p09)
    4. names of (usually multi-record consisting of multiple continuous segments) records are of the pattern "matched/pXX/pXXNNNN/pXXNNNN-YYYY-MM-DD-hh-mm" (one subdirectory "matched/pXX/pXXNNNN/" usually contains several), where "N", "X", "Y", "M", "D", "h", "m" are digits
    5. in most cases, the waveform (multi-)record is paired with a numerics record, which has the same name as the associated waveform record, with an "n" added to the end "pXXNNNN-YYYY-MM-DD-hh-mmn" ("n" is the letter, not a digit)
    6. in each folder "matched/pXX/pXXNNNN/", the files of patterns "3[\d]{6}_[\d]{4}.dat", "3[\d]{6}_[\d]{4}.hea", "3[\d]{6}n.dat", "3[\d]{6}_layout.hea" are from the parent `MIMIC3WDB`, with only the "3[\d]{6}.hea", "3[\d]{6}n.hea" files from the parent `MIMIC3WDB` replaced by "pXXNNNN-YYYY-MM-DD-hh-mm.hea", "pXXNNNN-YYYY-MM-DD-hh-mmn.hea" files correspondingly

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
    [4] https://physionet.org/content/mimic3wdb/1.0/
    [5] https://physionet.org/content/mimic3wdb-matched/1.0/
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
        super().__init__(db_name="mimic3wdb", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.metadata_files = ED(
            all_records=os.path.join(self.db_dir, "RECORDS"),
            waveforms=os.path.join(self.db_dir, "RECORDS-waveforms"),
            numerics=os.path.join(self.db_dir, "RECORDS-numerics"),
        )
        self.fs = None   # typically 125
        
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
