# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real
from ..utils import ArrayLike

from ..base import ImageDataBases


__all__ = [
    "CelebA"
]


class CelebA(ImageDataBases):
    """
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """
        """
        super().__init__(db_name="CelebA", db_path=db_path, verbose=verbose, **kwargs)
  