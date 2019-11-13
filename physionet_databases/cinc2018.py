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
    "CINC2018",
]


class CINC2018(PhysioNetDataBase):
    """ NOT Finished

    about CINC2018: to write

    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='challenge/2018/', db_path=db_path, **kwargs)


    def get_patient_id(self, rec) -> int:
        """

        """
        head = '2018'
        mid = rec[2:4]
        tail = rec[-4:]
        pid = int(head+mid+tail)
        return pid


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
