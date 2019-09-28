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

    ABOUT the database PPG_BP:
    1. the PPG sensor:
        1.1. sensor model was SEP9AF-2 (SMPLUS Company, Korea)
        1.2. contains dual LED with 660nm (Red light) and 905 nm (Infrared) wavelengths
        1.3. sampling rate 1 kHz and 12-bit ADC
        1.4. hardware filter design is 0.5‒12Hz bandpass
    more to be written

    References:
    -----------
    [1] Liang Y, Chen Z, Liu G, et al. A new, short-recorded photoplethysmogram dataset for blood pressure monitoring in China[J]. Scientific data, 2018, 5: 180020.
    [2] Allen J. Photoplethysmography and its application in clinical physiological measurement[J]. Physiological measurement, 2007, 28(3): R1.
    [3] Elgendi M. On the analysis of fingertip photoplethysmogram signals[J]. Current cardiology reviews, 2012, 8(1): 14-25.
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """ finished, to be improved,

        Parameters:
        -----------
        db_path: str,
            storage path of the database
        verbose: int, default 2,

        typical 'db_path':
        ------------------
        to be written
        """
        super().__init__(db_name="PPG_BP", db_path=db_path, verbose=verbose, **kwargs)

        self.ppg_data_path = None
        self.unkown_file = None
        self.ann_file = None
        self.form_paths()

        self.freq = 1000
        self.all_records = sorted(list(set([fn.split('_')[0] for fn in os.listdir(self.ppg_data_path)])), key=lambda r:int(r))
        self.rec_ext = '.txt'

        self.ann_items = [
            'Num.', 'subject_ID', 'Sex(M/F)', 'Age(year)', 'Height(cm)', 'Weight(kg)', 'BMI(kg/m^2)',
            'Systolic Blood Pressure(mmHg)', 'Diastolic Blood Pressure(mmHg)', 'Heart Rate(b/m)',
            'Hypertension', 'Diabetes', 'cerebral infarction', 'cerebrovascular disease',
        ]


    def form_paths(self) -> NoReturn:
        """ finished, checked, to be improved,

        """
        self.ppg_data_path = self.db_path + '0_subject/'
        self.unkown_file = self.db_path + 'Table 1.xlsx'
        self.ann_file = self.db_path + 'PPG-BP dataset.xlsx'


    def get_patient_id(self, rec_no:int) -> int:
        """ not finished,

        Parameters:
        -----------
        rec_no: int,
            number of the record, or 'subject_ID'

        Returns:
        int, the `patient_id` corr. to `rec_no`
        """
        return 0
    

    def database_info(self, detailed:bool=False) -> NoReturn:
        """ not finished,

        print the information about the database

        detailed: bool, default False,
            if False, an short introduction of the database will be printed,
            if True, then docstring of the class will be printed additionally
        """
        raw_info = {}

        print(raw_info)
        
        if detailed:
            print(self.__doc__)
        

    def load_ppg_data(self, rec_no:int, seg_no:int) -> np.ndarray:
        """ finished, checked,

        Parameters:
        -----------
        rec_no: int,
            number of the record, or 'subject_ID'
        seg_no: int,
            number of the segment measured from the subject
        
        Returns:
        --------
        ndarray, the ppg data
        """
        rec_fn = "{}_{}.txt".format(self.all_records[rec_no], seg_no)
        data = []
        with open(self.ppg_data_path+rec_fn, 'r') as f:
            data = f.readlines()
        data = np.array([float(i) for i in data[0].split('\t') if len(i)>0]).astype(int)
        
        if self.verbose >= 2:
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(figsize=(8,4))
            ax.plot(np.arange(0,len(data)/freq,1/freq),data)
            plt.show()
        
        return data


    def load_ann(self, rec_no:Optional[int]=None) -> pd.DataFrame:
        """ finished, checked,
        
        Parameters:
        -----------
        rec_no: int, optional,
            number of the record, or 'subject_ID',
            if not specified, then all annotations will be returned
        
        Returns:
        --------
        DataFrame, the annotations
        """
        df_ann = pd.read_excel(self.ann_file)
        df_ann.columns = df_ann.iloc[0]
        df_ann = df_ann[1:].reset_index(drop=True)
        
        if rec_no is None:
            return df_ann
        
        df_ann = df_ann[df_ann['subject_ID']==int(self.all_records[rec_no])].reset_index(drop=True)
        return df_ann


    def load_diagnosis(self, rec_no:int) -> Union[List[str],list]:
        """ finished, checked,
        
        Parameters:
        -----------
        rec_no: int,
            number of the record, or 'subject_ID'
        
        Returns:
        --------
        list, the list of diagnosis or empty list for the normal subjects
        """
        diagonosis_items = ['Hypertension', 'Diabetes', 'cerebral infarction', 'cerebrovascular disease']
        df_ann = load_ann(rec_no)[diagonosis_items].dropna(axis=1)
        diagonosis = [item for item in df_ann.iloc[0].tolist() if item != 'Normal']
        return diagonosis
