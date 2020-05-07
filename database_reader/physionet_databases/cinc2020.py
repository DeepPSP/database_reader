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
    "CINC2020",
]


class CINC2020(PhysioNetDataBase):
    """ NOT Finished,

    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020

    ABOUT CINC2020:
    ---------------
    1. 

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------

    References:
    -----------
    [1] https://physionetchallenges.github.io/2020/
    """
    def __init__(self, db_path:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_path: str, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='CINC2020', db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = None
        