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

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
    get_record_list_recursive3,
)
from ..base import PhysioNetDataBase


__all__ = [
    "CINC2017",
]


class CINC2017(PhysioNetDataBase):
    """ finished, NOT checked,

    AF Classification from a Short Single Lead ECG Recording
    - The PhysioNet Computing in Cardiology Challenge 2017

    ABOUT CINC2017:
    ---------------
    1. training set contains 8,528 single lead ECG recordings lasting from 9 s to just over 60 s, and the test set contains 3,658 ECG recordings of similar lengths
    2. records are of frequency 300 Hz and have been band pass filtered
    3. data distribution:
        Type	        	                    Time length (s)
                        # recording     Mean	SD	    Max	    Median	Min
        Normal	        5154	        31.9	10.0	61.0	30	    9.0
        AF              771             31.6	12.5	60	    30	    10.0
        Other rhythm	2557	        34.1	11.8	60.9	30	    9.1
        Noisy	        46	            27.1	9.0	    60	    30	    10.2
        Total	        8528	        32.5	10.9	61.0	30	    9.0

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. atrial fibrillation (AF) detection

    References:
    -----------
    [1] https://physionet.org/content/challenge-2017/1.0.0/
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """ finished, checked,

        Parameters:
        -----------
        db_dir: str, optional,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="CINC2017", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 300
        
        self.rec_ext = "mat"
        self.ann_ext = "hea"

        self._all_records = []
        self._ls_rec()

        self._df_ann = pd.read_csv(os.path.join(self.db_dir, "REFERENCE.csv"), header=None)
        self._df_ann.columns = ["rec", "ann"]
        self._df_ann_ori = pd.read_csv(os.path.join(self.db_dir, "REFERENCE-original.csv")header=None)
        self._df_ann_ori.columns = ["rec", "ann"]
        # ["N", "A", "O", "~"]
        self._all_ann = list(set(self._df_ann.ann.unique().tolist() + self._df_ann_ori.ann.unique().tolist()))
        self.d_ann_names = {
            "N": "Normal",
            "A": "AF",
            "O": "Other rhythm",
            "~": "Noisy",
        }


    def _ls_rec(self) -> NoReturn:
        """
        """
        fp = os.path.join(self.db_dir, "RECORDS")
        if os.path.isfile(fp):
            with open(fp, "r") as f:
                self._all_records = f.read().splitlines()
                return
        self._all_records = get_record_list_recursive3(
            db_dir=self.db_dir,
            rec_patterns=f"A[\d]{{5}}.{self.rec_ext}"
        )
        with open(fp, "w") as f:
            for rec in self._all_records:
                f.write(f"{rec}\n")


    def get_subject_id(self, rec) -> int:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        pid: int,
            the `subject_id` corr. to `rec`
        """
        raise NotImplementedError


    def load_data(self, rec:str, data_format:str="channel_first", units:str="mV") -> np:ndarray:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record
        data_format: str, default "channel_first",
            format of the ecg data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (of dimension 1, without channel dimension)
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"

        Returns:
        --------
        data: ndarray,
            data loaded from `rec`, with given units and format
        """
        wr = wfdb.rdrecord(os.path.join(self.db_dir, rec))
        data = wr.p_signal

        if wr.units[0].lower() == units.lower():
            pass
        elif wr.units[0].lower() in ["uv", "μv"] and units.lower() == "mv":
            data = data / 1000
        elif units.lower() in ["uv", "μv"] and wr.units[0].lower() == "mv":
            data = data * 1000

        data = data.squeeze()
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data[np.newaxis,...]
        elif data_format.lower() in ["channel_last", "lead_last"]:
            data = data[...,np.newaxis]
        return data


    def load_ann(self, rec:str, original:bool=False) -> str:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record
        original: bool, default False,
            if True, load annotations from the file `REFERENCE-original.csv`,
            otherwise from `REFERENCE.csv`

        Returns:
        --------
        ann: str,
            annotation (label) of the record
        """
        assert rec in self.all_records
        if original:
            df = self._df_ann_ori
        else:
            df = self._df_ann
        row = df[df.ann==rec].iloc[0]
        ann = row.ann
        return ann


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
