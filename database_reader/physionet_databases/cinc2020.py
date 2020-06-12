# -*- coding: utf-8 -*-
"""
"""
import os
import wfdb
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from database_reader.utils import (
    ArrayLike,
    AF, I_AVB, LBBB, RBBB, PAC, PVC, STD, STE,
    Dx_map,
)
from database_reader.base import PhysioNetDataBase
from database_reader.other_databases import CPSC2018


__all__ = [
    "CINC2020",
]


class CINC2020(PhysioNetDataBase):
    """ NOT Finished,

    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020

    ABOUT CINC2020:
    ---------------
    0. There are 6 difference resources of training data, listed as follows:
        A. 6,877 recordings from China Physiological Signal Challenge in 2018 (CPSC2018):  https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz 
        B. 3,453 recordings from China 12-Lead ECG Challenge Database (unused data from CPSC2018 and NOT the CPSC2018 test data): https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_2.tar.gz
        C. 74 recordings from the St Petersburg INCART 12-lead Arrhythmia Database: https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_StPetersburg.tar.gz 
        D. 516 recordings from the PTB Diagnostic ECG Database: https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB.tar.gz
        E. 21,837 recordings from the PTB-XL electrocardiography Database: https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_PTB-XL.tar.gz
        F. 10,344 recordings from a Georgia 12-Lead ECG Challenge Database: https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_E.tar.gz
    In total, 43,101 labeled recordings of 12-lead ECGs from four countries (China, Germany, Russia, and the USA) across 3 continents have been posted publicly for this Challenge, with approximately the same number hidden for testing, representing the largest public collection of 12-lead ECGs

    1. the first part training data comes from CPSC2018, whose folder name is `Training_WFDB`. For this part, ref. the docstring of `database_reader.other_databases.cpsc2018.CPSC2018`

    2. the second part training data have folder name `Training_2`
    3. the second part training data are in the `channel first` format
    4. there are some errors or debatable labels in all of the data in the second part of training data
    5. for the second part of data, leads can be inverted, noisy, mislabeled. The organizers have deliberately made no attempt to clean this up. The test data contains better labels, but it is not perfect either, and although it roughly corresponds to the training data, it includes some deliberate differences (from google group)
    6. the second part contains 3453 records, whose file names start with 'Q'. The numbering is not continuous, with gaps. The last record is 'Q3581'
    7. annotation structures of the second part are the same with the first part
    8.

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. ECG arrhythmia detection

    References:
    -----------
    [1] https://physionetchallenges.github.io/2020/
    [2] database_reader.other_databases.cpsc2018
    """
    def __init__(self, db_path:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_path: str, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name='CINC2020', db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
        self.freq = 500
        self.spacing = 1000 / self.freq
        self.first_part_dir = os.path.join(db_path, "Training_WFDB")
        self.second_part_dir = os.path.join(db_path, "Training_2")
        self.rec_ext = '.mat'
        self.ann_ext = '.hea'

        self.all_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',]
        self.all_diagnosis = ['N', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE',]
        self.all_diagnosis_original = sorted(['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE',])
        self.diagnosis_abbr_to_full = {
            'N': 'Normal',
            'AF': 'Atrial fibrillation',
            'I-AVB': 'First-degree atrioventricular block',
            'LBBB': 'Left bundle brunch block',
            'RBBB': 'Right bundle brunch block',
            'PAC': 'Premature atrial contraction',
            'PVC': 'Premature ventricular contraction',
            'STD': 'ST-segment depression',
            'STE': 'ST-segment elevated',
        }


        self.first_part_reader = CPSC2018(db_path=self.first_part_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.second_part_reader = CPSC2018(db_path=self.second_part_dir, working_dir=working_dir, verbose=verbose, **kwargs)


    def get_patient_id(self, rec:str) -> int:
        """ not finished,

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        pid: int,
            the `patient_id` corr. to `rec_no`
        """
        pid = 0
        raise NotImplementedError


    def database_info(self, detailed:bool=False) -> NoReturn:
        """ not finished,

        print the information about the database

        Parameters:
        -----------
        detailed: bool, default False,
            if False, an short introduction of the database will be printed,
            if True, then docstring of the class will be printed additionally
        """
        raw_info = {}

        print(raw_info)
        
        if detailed:
            print(self.__doc__)


    def load_data(self, rec:str, data_format='channels_last') -> np.ndarray:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record
        data_format: str, default 'channels_last',
            format of the ecg data, 'channels_last' or 'channels_first' (original)
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        if rec.startswith("A"):
            data = self.first_part_reader.load_data(rec, data_format)
        elif rec.startswith("Q"):
            data = self.second_part_reader.load_data(rec, data_format)
        return data

    
    def load_ann(self, rec:str, keep_original:bool=False) -> dict:
        """ finished, checked,
        
        Parameters:
        -----------
        rec: str,
            name of the record
        keep_original: bool, default False,
            keep the original annotations or not,
            mainly concerning 'N' and 'Normal'
        
        Returns:
        --------
        ann_dict, dict,
            the annotations with items: ref. self.ann_items
        """
        if rec.startswith("A"):
            ann_dict = self.first_part_reader.load_ann(rec_no, keep_original)
        elif rec.startswith("Q"):
            ann_dict = self.second_part_reader.load_ann(rec_no, keep_original)
        return ann_dict
