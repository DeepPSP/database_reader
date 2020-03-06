# -*- coding: utf-8 -*-
import os
import pprint
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from utils import ArrayLike
from base import PhysioNetDataBase


__all__ = [
    "CINC2020",
]


class CINC2020(PhysioNetDataBase):
    """ NOT Finished,

    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020

    ABOUT CINC2020:
    ---------------
    1. 

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------

    References:
    -----------
    [1] https://physionetchallenges.github.io/2020/
    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='challenge/2020/', db_path=db_path, **kwargs)
        self.freq = None
        