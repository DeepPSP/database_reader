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
    "BIDMC",
]


class BIDMC(PhysioNetDataBase):
    """ NOT finished,

    BIDMC PPG and Respiration Dataset

    about bidmc: to write

    References:
    -----------
    [1] https://physionet.org/content/bidmc/1.0.0/
    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='bidmc', db_path=db_path, **kwargs)
        try:
            self.all_records = wfdb.get_record_list('bidmc')
        except:
            try:
                self.all_records = os.listdir(self.db_path)
                self.all_records = list(set([item.split('.')[0] for item in self.all_records]))
            except:
                self.all_records = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05', 'bidmc06', 'bidmc07', 'bidmc08', 'bidmc09', 'bidmc10', 'bidmc11', 'bidmc12', 'bidmc13', 'bidmc14', 'bidmc15', 'bidmc16', 'bidmc17', 'bidmc18', 'bidmc19', 'bidmc20', 'bidmc21', 'bidmc22', 'bidmc23', 'bidmc24', 'bidmc25', 'bidmc26', 'bidmc27', 'bidmc28', 'bidmc29', 'bidmc30', 'bidmc31', 'bidmc32', 'bidmc33', 'bidmc34', 'bidmc35', 'bidmc36', 'bidmc37', 'bidmc38', 'bidmc39', 'bidmc40', 'bidmc41', 'bidmc42', 'bidmc43', 'bidmc44', 'bidmc45', 'bidmc46', 'bidmc47', 'bidmc48', 'bidmc49', 'bidmc50', 'bidmc51', 'bidmc52', 'bidmc53']


    def get_patient_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
