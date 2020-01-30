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
    "CAPSLPDB",
]


class CAPSLPDB(PhysioNetDataBase):
    """ NOT finished,

    CAP Sleep Database

    About capslpdb: to write

    References:
    -----------
    [1] https://physionet.org/content/capslpdb/1.0.0/
    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='capslpdb', db_path=db_path, **kwargs)


    def get_patient_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
