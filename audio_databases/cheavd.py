# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, List, NoReturn

from base import AudioDataBases


__all__ = [
    "CHEAVD"
]


class CHEAVD(AudioDataBases):
    """
    Chinese Natural Emotional Audio–Visual Database
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """

        """
        super().__init__(db_name="CASIA_CHEAVD", db_path=db_path, verbose=verbose, **kwargs)
