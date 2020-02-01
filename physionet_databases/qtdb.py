# -*- coding: utf-8 -*-
import os
import pprint
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real
from ..utils import ArrayLike

from ..base import PhysioNetDataBase


__all__ = [
    "QTDB",
]


class QTDB(PhysioNetDataBase):
    """ NOT finished,

    QT Database

    ABOUT qtdb:
    -----------
    1. contains 105 fifteen-minute two-lead ECG recordings
    2. contains onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. ecg wave delineation
    2. ST segment

    References:
    -----------
    [1] https://www.physionet.org/content/qtdb/1.0.0/
    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='qtdb', db_path=db_path, **kwargs)
        try:
            self.all_records = wfdb.get_record_list('qtdb')
        except:
            try:
                self.all_records = os.listdir(self.db_path)
                self.all_records = list(set([os.path.splitext(item)[0] for item in self.all_records]))
            except:
                self.all_records = ['sel100', 'sel102', 'sel103', 'sel104', 'sel114', 'sel116', 'sel117', 'sel123', 'sel14046', 'sel14157', 'sel14172', 'sel15814', 'sel16265', 'sel16272', 'sel16273', 'sel16420', 'sel16483', 'sel16539', 'sel16773', 'sel16786', 'sel16795', 'sel17152', 'sel17453', 'sel213', 'sel221', 'sel223', 'sel230', 'sel231', 'sel232', 'sel233', 'sel30', 'sel301', 'sel302', 'sel306', 'sel307', 'sel308', 'sel31', 'sel310', 'sel32', 'sel33', 'sel34', 'sel35', 'sel36', 'sel37', 'sel38', 'sel39', 'sel40', 'sel41', 'sel42', 'sel43', 'sel44', 'sel45', 'sel46', 'sel47', 'sel48', 'sel49', 'sel50', 'sel51', 'sel52', 'sel803', 'sel808', 'sel811', 'sel820', 'sel821', 'sel840', 'sel847', 'sel853', 'sel871', 'sel872', 'sel873', 'sel883', 'sel891', 'sele0104', 'sele0106', 'sele0107', 'sele0110', 'sele0111', 'sele0112', 'sele0114', 'sele0116', 'sele0121', 'sele0122', 'sele0124', 'sele0126', 'sele0129', 'sele0133', 'sele0136', 'sele0166', 'sele0170', 'sele0203', 'sele0210', 'sele0211', 'sele0303', 'sele0405', 'sele0406', 'sele0409', 'sele0411', 'sele0509', 'sele0603', 'sele0604', 'sele0606', 'sele0607', 'sele0609', 'sele0612', 'sele0704']
        self.freq = 250
        self.all_extensions = ['atr', 'man', 'q1c', 'q2c', 'qt1', 'qt2', 'pu', 'pu0', 'pu1']
        """
        1. .atr:    reference beat annotations from original database (not available in all cases)
        2. .man:    reference beat annotations for selected beats only
        3. .q1c:    manually determined waveform boundary measurements for selected beats (annotator 1 only -- second pass)
        4. .q2c:    manually determined waveform boundary measurements for selected beats (annotator 2 only -- second pass; available for only 11 records)
        5. .q1t:    manually determined waveform boundary measurements for selected beats (annotator 1 only -- first pass)
        6. .q2t:    manually determined waveform boundary measurements for selected beats (annotator 2 only -- first pass; available for only 11 records)
        7. .pu:     automatically determined waveform boundary measurements for all beats (based on both signals)
        8. .pu0:    automatically determined waveform boundary measurements for all beats (based on signal 0 only)
        9. .pu1:    automatically determined waveform boundary measurements for all beats (based on signal 1 only)
        """
        self.all_leads = ['MLII', 'V4-V5', 'ECG2', 'ML5', 'D4', 'V1-V2', 'V5', 'V2', 'D3', 'ECG1', 'V3', 'V2-V3', 'CM5', 'CM4', 'V4', 'CC5', 'CM2', 'V1', 'mod.V1']
        self.all_annotations = ['(', ')', 'N', 't', 'p']


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
