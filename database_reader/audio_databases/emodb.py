# -*- coding: utf-8 -*-
import os
from typing import Union, Optional, Any, List, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import AudioDataBase


__all__ = [
    "EmoDB"
]


class EmoDB(AudioDataBase):
    """
    Berlin Database of Emotional Speech

    References：
    -----------
    [1] http://emodb.bilderbar.info/index-1024.html
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="Berlin_EmoDB", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
