# -*- coding: utf-8 -*-
"""
"""
import os
import json
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, NoReturn
from numbers import Real

import numpy as np
import pandas as pd
import wfdb

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "AFDB",
]


class AFDB(PhysioNetDataBase):
    """ NOT Finished,

    MIT-BIH Atrial Fibrillation Database

    ABOUT afdb:
    -----------

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. 

    References:
    -----------
    [1] https://physionet.org/content/afdb/1.0.0/
    [2] Moody GB, Mark RG. A new method for detecting atrial fibrillation using R-R intervals. Computers in Cardiology. 10:227-230 (1983).
    """
    def __init__(self, db_dir:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='afdb', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        # self.freq = 100
        # self.data_ext = "dat"
        # self.ann_ext = "apn"

        self._ls_rec()

        raise NotImplementedError
