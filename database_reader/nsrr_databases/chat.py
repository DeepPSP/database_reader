# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Iterable, Sequence, NoReturn
from numbers import Real

from database_reader.utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from database_reader.base import NSRRDataBase


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
    def __init__(self, db_path:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_path: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='CHAT', db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
        