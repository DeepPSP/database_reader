# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Iterable, Sequence, NoReturn
from numbers import Real
from ..utils import ArrayLike

from ..base import NSRRDataBase


__all__ = [
    "CHAT",
]


class CHAT(NSRRDataBase):
    """

    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        """

        """
        super().__init__(db_name='chat', db_path=db_path, **kwargs)
        