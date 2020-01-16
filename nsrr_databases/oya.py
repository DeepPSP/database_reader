# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import os
import numpy as np
import pandas as pd
import pprint
import xmltodict as xtd
from pyedflib import EdfReader
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Iterable, Sequence, NoReturn
from numbers import Real
from ..utils import ArrayLike
from utils.utils_interval import intervals_union

from ..base import NSRRDataBase


__all__ = [
    "OYA",
]


class OYA(NSRRDataBase):
    """
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """
        """
        super().__init__(db_name='OYA', db_path=db_path, verbose=verbose, **kwargs)


    def get_patient_id(self,):
        """ not finished,

        Parameters:
        -----------
        rec: str,
            record name, typically in the form ...

        Returns:
        --------
        pid, int, `patient_id` derived from `rec`
        """
        raise NotImplementedError


    def form_paths(self) -> NoReturn:
        """ not finished,

        """
        # self.psg_data_path = os.path.join(self.db_path, "polysomnography", "edfs")
        # self.ann_path = os.path.join(self.db_path, "datasets")
        # self.hrv_ann_path = os.path.join(self.db_path, "hrv-analysis")
        # self.eeg_ann_path = os.path.join(self.db_path, "eeg-spectral-analysis")
        # self.wave_deli_path = os.path.join(self.db_path, "polysomnography", "annotations-rpoints")
        # self.event_ann_path = os.path.join(self.db_path, "polysomnography", "annotations-events-nsrr")
        # self.event_profusion_ann_path = os.path.join(self.db_path, "polysomnography", "annotations-events-profusion")
    