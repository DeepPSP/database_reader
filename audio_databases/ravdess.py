# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, List, NoReturn

from base import AudioDataBases


__all__ = [
    "RAVDESS"
]


class RAVDESS(AudioDataBases):
    """
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """
        Ryerson Audio-Visual Database of Emotional Speech and Song
        """
        super().__init__(db_name="RAVDESS", db_path=db_path, verbose=verbose, **kwargs)
