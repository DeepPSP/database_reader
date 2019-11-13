# -*- coding: utf-8 -*-
import os
import pprint
import wfdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real
from ..utils import ArrayLike

from ..base import PhysioNetDataBase


__all__ = [
    "MIMIC3",
]


class MIMIC3(PhysioNetDataBase):
    """
    
    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        """

        """
        super().__init__(db_name='ltstdb', db_path=db_path, **kwargs)
