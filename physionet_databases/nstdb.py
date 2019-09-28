# -*- coding: utf-8 -*-
import os
import pprint
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real
from ..misc import ArrayLike

from ..base import PhysioNetDataBase


__all__ = [
    "NSTDB",
]


class NSTDB(PhysioNetDataBase):
    """ NOT finished,

    about qtdb: to write

    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='nstdb', db_path=db_path, **kwargs)
        try:
            self.all_records = wfdb.get_record_list('apnea-ecg')
        except:
            try:
                self.all_records = os.listdir(self.db_path)
                self.all_records = list(set([item.split('.')[0] for item in self.all_records]))
            except:
                self.all_records = []
        

    def get_patient_id(self, rec) -> int:
        """

        """
        return


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
