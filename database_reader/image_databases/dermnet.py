# -*- coding: utf-8 -*-
"""
"""
import os
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import ImageDataBase


__all__ = [
    "DermNet"
]


class DermNet(ImageDataBase):
    """
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
        super().__init__(db_name="DermNet", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
