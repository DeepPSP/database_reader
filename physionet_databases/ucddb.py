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
    "UCDDB",
]


class UCDDB(PhysioNetDataBase):
    """ NOT finished,

    St. Vincent's University Hospital / University College Dublin Sleep Apnea Database

    About ucddb: to write

    References:
    -----------
    [1] https://physionet.org/content/ucddb/1.0.0/
    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='ucddb', db_path=db_path, **kwargs)


    def get_patient_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
