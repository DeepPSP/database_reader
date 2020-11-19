# -*- coding: utf-8 -*-
"""
docstring, to write
"""
import os
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Iterable, Sequence, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import xmltodict as xtd
from pyedflib import EdfReader

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..utils.utils_universal import intervals_union
from ..base import NSRRDataBase


__all__ = [
    "OYA",
]


class OYA(NSRRDataBase):
    """

    One Year of Actigraphy

    ABOUT oya:
    ----------
    to write

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------

    References:
    -----------
    [1] https://sleepdata.org/datasets/oya
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="OYA", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)


    def get_subject_id(self, rec:str) -> int:
        """
        Attach a `subject_id` to the record, in order to facilitate further uses

        Parameters:
        -----------
        rec: str,
            record name

        Returns:
        --------
        int, a `subject_id` attached to the record `rec`
        """
        raise NotImplementedError


    def form_paths(self) -> NoReturn:
        """ not finished,

        """
        # self.psg_data_path = os.path.join(self.db_dir, "polysomnography", "edfs")
        # self.ann_path = os.path.join(self.db_dir, "datasets")
        # self.hrv_ann_path = os.path.join(self.db_dir, "hrv-analysis")
        # self.eeg_ann_path = os.path.join(self.db_dir, "eeg-spectral-analysis")
        # self.wave_deli_path = os.path.join(self.db_dir, "polysomnography", "annotations-rpoints")
        # self.event_ann_path = os.path.join(self.db_dir, "polysomnography", "annotations-events-nsrr")
        # self.event_profusion_ann_path = os.path.join(self.db_dir, "polysomnography", "annotations-events-profusion")
    