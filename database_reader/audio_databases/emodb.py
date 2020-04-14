# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, List, NoReturn

from database_reader.base import AudioDataBase


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
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """
        """
        super().__init__(db_name="Berlin_EmoDB", db_path=db_path, verbose=verbose, **kwargs)
