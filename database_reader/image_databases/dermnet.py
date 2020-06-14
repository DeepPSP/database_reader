# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from database_reader.utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from database_reader.base import ImageDataBase


__all__ = [
    "DermNet"
]


class DermNet(ImageDataBase):
    """
    """
    def __init__(self, db_path:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_path: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="DermNet", db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
