# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from database_reader.utils import ArrayLike
from database_reader.base import ImageDataBase


__all__ = [
    "DermNet"
]


class DermNet(ImageDataBase):
    """
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """
        """
        super().__init__(db_name="DermNet", db_path=db_path, verbose=verbose, **kwargs)
