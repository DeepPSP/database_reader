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
)
from ..base import PhysioNetDataBase


__all__ = [
    "CAPSLPDB",
]


class CAPSLPDB(PhysioNetDataBase):
    """ NOT finished,

    CAP Sleep Database

    ABOUT capslpdb:
    ---------------
    1. contains 108 polysomnographic (PSG) recordings, including 16 healthy subjects and 92 pathological recordings, in EDF format, NOT the usual wfdb .dat format
    2. The 92 pathological recordings include 40 recordings of patients diagnosed with nocturnal frontal lobe epilepsy (NFLE), 22 affected by REM behavior disorder (RBD), 10 with periodic leg movements (PLM), 9 insomniac, 5 narcoleptic, 4 affected by sleep-disordered breathing (SDB) and 2 by bruxism
    3. 

    NOTE:
    -----
    1. background knowledge aboute CAP:
    The Cyclic Alternating Pattern (CAP) is a periodic EEG activity occurring during NREM sleep. It is characterized by cyclic sequences of cerebral activation (phase A) followed by periods of deactivation (phase B) which separate two successive phase A periods with an interval <1 min. A phase A period and the following phase B period define a CAP cycle, and at least two CAP cycles are required to form a CAP sequence

    ISSUES:
    -------

    Usage:
    ------
    1. sleep stage
    1. sleep cyclic alternating pattern

    References:
    -----------
    [1] https://physionet.org/content/capslpdb/1.0.0/
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
        super().__init__(db_name='capslpdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.data_ext = "edf"
        self.ann_ext = "st"
        self.alias_ann_ext = "txt"
        self.freq = None  # psg data with different frequencies for each signal
        
        self._ls_rec()


    def _ls_rec(self, local:bool=True) -> NoReturn:
        """ finished, checked,

        find all records (relative path without file extension),
        and save into `self._all_records` for further use

        Parameters:
        -----------
        local: bool, default True,
            if True, read from local storage, prior to using `wfdb.get_record_list`
        """
        try:
            super()._ls_rec(local=local)
        except:
            self._all_records = ['brux1', 'brux2', 'ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'narco1', 'narco2', 'narco3', 'narco4', 'narco5', 'nfle10', 'nfle11', 'nfle12', 'nfle13', 'nfle14', 'nfle15', 'nfle16', 'nfle17', 'nfle18', 'nfle19', 'nfle1', 'nfle20', 'nfle21', 'nfle22', 'nfle23', 'nfle24', 'nfle25', 'nfle26', 'nfle27', 'nfle28', 'nfle29', 'nfle2', 'nfle30', 'nfle31', 'nfle32', 'nfle33', 'nfle34', 'nfle35', 'nfle36', 'nfle37', 'nfle38', 'nfle39', 'nfle3', 'nfle40', 'nfle4', 'nfle5', 'nfle6', 'nfle7', 'nfle8', 'nfle9', 'plm10', 'plm1', 'plm2', 'plm3', 'plm4', 'plm5', 'plm6', 'plm7', 'plm8', 'plm9', 'rbd10', 'rbd11', 'rbd12', 'rbd13', 'rbd14', 'rbd15', 'rbd16', 'rbd17', 'rbd18', 'rbd19', 'rbd1', 'rbd20', 'rbd21', 'rbd22', 'rbd2', 'rbd3', 'rbd4', 'rbd5', 'rbd6', 'rbd7', 'rbd8', 'rbd9', 'sdb1', 'sdb2', 'sdb3', 'sdb4']


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
