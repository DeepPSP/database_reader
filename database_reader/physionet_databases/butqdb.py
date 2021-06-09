# -*- coding: utf-8 -*-
"""
"""
import os
import json
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import wfdb

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "BUTQDB",
]


class BUTQDB(PhysioNetDataBase):
    """ NOT Finished, 

    Brno University of Technology ECG Quality Database

    ABOUT ludb
    ----------
    1. 

    NOTE
    ----

    ISSUES
    ------

    Usage
    -----
    1. ECG quality estimation

    References
    ----------
    [1] https://physionet.org/content/butqdb/1.0.0/
    [2] Nemcova, A., Smisek, R., Opravilová, K., Vitek, M., Smital, L., & Maršánová, L. (2020). Brno University of Technology ECG Quality Database (BUT QDB) (version 1.0.0). PhysioNet. https://doi.org/10.13026/kah4-0w24.
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs:Any) -> NoReturn:
        """
        Parameters
        ----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments
        """
        super().__init__(db_name="butqdb", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.fs = 500
        self.data_ext = "dat"
        self.ann_ext = "csv"

        self._ls_rec()


    def _ls_rec(self, local:bool=True) -> NoReturn:
        """
        """
        raise NotImplementedError
    

    def get_subject_id(self, rec:str) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)


    def load_acc_data(self, rec:str) -> np.ndarray:
        """

        """
        raise NotImplementedError
