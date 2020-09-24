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
    1. training set contains 8,528 single lead ECG recordings lasting from 9 s to just over 60 s, and the test set contains 3,658 ECG recordings of similar lengths
    2. records are of frequency 300 Hz and have been band pass filtered
    3. data distribution:
        Type	        	                    Time length (s)
                        # recording     Mean	SD	    Max	    Median	Min
        Normal	        5154	        31.9	10.0	61.0	30	    9.0
        AF              771             31.6	12.5	60	    30	    10.0
        Other rhythm	2557	        34.1	11.8	60.9	30	    9.1
        Noisy	        46	            27.1	9.0	    60	    30	    10.2
        Total	        8528	        32.5	10.9	61.0	30	    9.0

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
        self.freq = 300
        self.rec_ext = 'mat'
        self.ann_ext = 'hea'
        self._ls_rec()


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
