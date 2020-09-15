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

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "LTAFDB",
]


class AFTDB(PhysioNetDataBase):
    """ NOT Finished,

    Long Term AF Database

    ABOUT ltafdb:
    -------------

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
        super().__init__(db_name='ltftdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        # self.freq = 100
        # self.data_ext = "dat"
        # self.ann_ext = "apn"

        self._ls_rec()

        raise NotImplementedError


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
