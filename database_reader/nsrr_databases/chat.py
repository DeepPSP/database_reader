# -*- coding: utf-8 -*-
"""
docstring, to write
"""
import os
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Iterable, Sequence, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import NSRRDataBase


__all__ = [
    "CHAT",
]


class CHAT(NSRRDataBase):
    """

    Childhood Adenotonsillectomy Trial

    ABOUT chat:
    -----------
    to write

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------

    References:
    -----------
    [1] https://sleepdata.org/datasets/chat
    """
    def __init__(self, db_dir:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='CHAT', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        