# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import os
import numpy as np
import pandas as pd
import xmltodict as xtd
from pyedflib import EdfReader
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Iterable, Sequence, NoReturn
from numbers import Real

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..utils.utils_universal import intervals_union
from ..base import NSRRDataBase


__all__ = [
    "MESA",
]


class MESA(NSRRDataBase):
    """

    Multi-Ethnic Study of Atherosclerosis

    ABOUT mesa:
    -----------
    ***ABOUT the dataset:
    1. subjects: 6,814 black, white, Hispanic, and Chinese-American men and women initially ages 45-84 at baseline in 2000-2002
    2. there have been four follow-up exams to date, in the years
        2003-2004,
        2004-2005,
        2005-2007,
        2010-2011
    3. between 2010-2012, 2,237 participants also were enrolled in a Sleep Exam (MESA Sleep) which included 
        full overnight unattended polysomnography (PSG),
        7-day wrist-worn actigraphy (ACT),
        a sleep questionnaire

    ***ABOUT ACT data:
    4. 2,237 participants, between 2010 and 2013
    5. epoch-by-epoch (EBE) data files (CSV) have been created for 2,159 participants with actigraphy data
    6. EBE data files have 15 columns, ref `self.actigraph_cols`
    7. sleep in ACTIVE intervals is never counted toward overall sleep totals. Actual sleep is tallied within REST intervals only. REST-S intervals indicate the period between sleep onset and offset

    ***ABOUT PSG data:
    8. signals contained:
        ECG
        cortical electroencephalograms (EEG): central C4-M1, occipital Oz-Cz, frontal Fz-Cz leads
        bilateral electrooculograms (EOG)
        chin EMG
        thoracic and abdominal respiratory inductance plethysmography
        airflow
        leg movements
        finger pulse oximetry
    9. 

    NOTE:
    -----
    1. actigraph data: epochs without activity counts and/or light readings typically indicate offwrist time and/or the watch's failure to provide a valid measurement for that epoch. These epochs should be excluded from study
    2. 

    ISSUES:
    -------

    Usage:
    ------
    1. 

    References:
    -----------
    [1] https://sleepdata.org/datasets/mesa
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
        super().__init__(db_name='MESA', db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)

        self.actigraph_cols = [
            "mesaid",
            "line",  # epoch line number
            "linetime",  # clock time, in the format of "HH:MM:SS", or "%H:%M:%S"
            "offwrist",  # off-wrist indicator, 
            "activity",  # Activity count
            "marker",  # event marker indicator
            "whitelight",  # in lux
            "redlight",  # in microwatts per square centimeter
            "greenlight",  # in microwatts per square centimeter
            "bluelight",  # in microwatts per square centimeter
            "wake",  # awake indicator, created by algorithm
            "interval",  # interval type: "ACTIVE", "REST", "REST-S"
            "dayofweek",  # 1 = Sunday / 2 = Monday / etc.
            "daybymidnight",
            "daybynoon",
        ]


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

    
    def database_info(self, detailed:bool=False) -> NoReturn:
        """ finished,

        print information about the database

        Parameters:
        -----------
        detailed: bool, default False,
            if False, "What","Who","When","Funding" will be printed,
            if True, then docstring of the class will be printed additionally
        """
        raw_info = {
            "What": "Multi-center longitudinal investigation of factors associated with cardiovascular disease",
            "Who": "6,814 black, white, Hispanic, and Chinese-American men and women",
            "When": "2000 to present",
            "Funding": "National Heart, Lung, and Blood Institute"
        }

        print(raw_info)
        
        if detailed:
            print(self.__doc__)
    