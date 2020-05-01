# -*- coding: utf-8 -*-
import os
import pprint
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from database_reader.utils import ArrayLike
from database_reader.base import PhysioNetDataBase


__all__ = [
    "CAPSLPDB",
]


class CAPSLPDB(PhysioNetDataBase):
    """ NOT finished,

    CAP Sleep Database

    ABOUT capslpdb:
    ---------------
    1. contains 108 polysomnographic (PSG) recordings, including 16 healthy subjects and 92 pathological recordings.
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
    def __init__(self, db_path:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        """
        super().__init__(db_name='capslpdb', db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = None  # psg data with different frequencies for each signal
        try:
            self.all_records = [os.path.splitext(item)[0] for item in wfdb.get_record_list('capslpdb')]
        except:
            # try:
            #     self.all_records = os.listdir(self.db_path)
            #     self.all_records = list(set([os.path.splitext(item)[0] for item in self.all_records]))
            # except:
            self.all_records = ['brux1', 'brux2', 'ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'narco1', 'narco2', 'narco3', 'narco4', 'narco5', 'nfle10', 'nfle11', 'nfle12', 'nfle13', 'nfle14', 'nfle15', 'nfle16', 'nfle17', 'nfle18', 'nfle19', 'nfle1', 'nfle20', 'nfle21', 'nfle22', 'nfle23', 'nfle24', 'nfle25', 'nfle26', 'nfle27', 'nfle28', 'nfle29', 'nfle2', 'nfle30', 'nfle31', 'nfle32', 'nfle33', 'nfle34', 'nfle35', 'nfle36', 'nfle37', 'nfle38', 'nfle39', 'nfle3', 'nfle40', 'nfle4', 'nfle5', 'nfle6', 'nfle7', 'nfle8', 'nfle9', 'plm10', 'plm1', 'plm2', 'plm3', 'plm4', 'plm5', 'plm6', 'plm7', 'plm8', 'plm9', 'rbd10', 'rbd11', 'rbd12', 'rbd13', 'rbd14', 'rbd15', 'rbd16', 'rbd17', 'rbd18', 'rbd19', 'rbd1', 'rbd20', 'rbd21', 'rbd22', 'rbd2', 'rbd3', 'rbd4', 'rbd5', 'rbd6', 'rbd7', 'rbd8', 'rbd9', 'sdb1', 'sdb2', 'sdb3', 'sdb4']


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
