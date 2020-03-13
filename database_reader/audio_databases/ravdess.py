# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, List, NoReturn

from database_reader.base import AudioDataBases


__all__ = [
    "RAVDESS"
]


class RAVDESS(AudioDataBases):
    """
    Ryerson Audio-Visual Database of Emotional Speech and Song

    References：
    -----------
    [1] https://zenodo.org/record/1188976#.XmCugqgzY2w
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """
        
        """
        super().__init__(db_name="RAVDESS", db_path=db_path, verbose=verbose, **kwargs)
