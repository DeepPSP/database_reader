# -*- coding: utf-8 -*-
"""
"""
import os
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "CINC2017",
]


class CINC2017(PhysioNetDataBase):
    """ NOT Finished,

    AF Classification from a Short Single Lead ECG Recording
    - The PhysioNet Computing in Cardiology Challenge 2017

    ABOUT CINC2017:
    ---------------
    1. 

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. atrial fibrillation (AF) detection

    References:
    -----------
    [1] https://physionet.org/content/challenge-2017/1.0.0/
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str, optional,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='CINC2017', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        raise NotImplementedError


    def get_subject_id(self, rec) -> int:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        pid: int,
            the `subject_id` corr. to `rec`
        """
        raise NotImplementedError


    def load_data(self,) -> np:ndarray:
        """
        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
