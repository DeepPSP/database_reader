# -*- coding: utf-8 -*-
import os
import pprint
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real
from ..misc import ArrayLike

from ..base import OtherDataBase


__all__ = [
    "PPGBP",
]


class PPGBP(OtherDataBase):
    """

    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        super().__init__(db_name="PPG_BP", db_path=db_path, verbose=verbose, **kwargs)
