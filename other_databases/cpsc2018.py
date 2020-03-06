# -*- coding: utf-8 -*-
import io
import os
import pprint
import wfdb
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from utils import ArrayLike
from base import OtherDataBase


__all__ = [
    "CPSC2018",
]


class CPSC2018(OtherDataBase):
    """

    The China Physiological Signal Challenge 2018:
    Automatic identification of the rhythm/morphology abnormalities in 12-lead ECGs

    ABOUT CPSC2018:
    ---------------
    1. training set contains 6,877 (female: 3178; male: 3699) 12 leads ECG recordings lasting from 6 s to just 60 s
    2. ECG recordings were sampled as 500 Hz
    3. the training data can be downloaded using links in Ref.[1], but the link in Ref.[2] is recommended. File structure will be assumed to follow Ref.[2]
    4. types of abnormal rhythm/morphology + normal in the training set:
            name                                    abbr.       number of records
        (0) Normal                                  N           918
        (1) Atrial fibrillation                     AF          1098
        (2) First-degree atrioventricular block     I-AVB       704
        (3) Left bundle brunch block                LBBB        207
        (4) Right bundle brunch block               RBBB        1695
        (5) Premature atrial contraction            PAC         556
        (6) Premature ventricular contraction       PVC         672
        (7) ST-segment depression                   STD         825
        (8) ST-segment elevated                     STE         202
    5. meanings in the .hea files: to write

    NOTE:
    -----
    

    ISSUES:
    -------

    Usage:
    ------

    References:
    -----------
    [1] http://2018.icbeb.org/#
    [1] https://physionetchallenges.github.io/2020/
    """
    def __init__(self, db_path:str, verbose:int=2, **kwargs):
        """ finished, to be improved,

        Parameters:
        -----------
        db_path: str,
            storage path of the database
        verbose: int, default 2,
        """
        super().__init__(db_name="CPSC2018", db_path=db_path, verbose=verbose, **kwargs)

        self.freq = 500
        self.rec_ext = '.mat'
        self.ann_ext = '.hea'
        self.all_records = [os.path.splitext(os.path.basename(item))[0] for item in glob.glob(os.path.join(db_path, '*'+self.rec_ext))]
        self.nb_records = 6877
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

        self.ann_items = [
            'rec_name',
            'nb_leads',
            'freq',
            'nb_samples',
            'datetime',
            'age',
            'sex',
            'diagnosis',
            'medical_prescription',
            'history',
            'symptom_or_surgery',
            'df_leads',
        ]


    def get_patient_id(self, rec_no:int) -> int:
        """ not finished,

        Parameters:
        -----------
        rec_no: int,
            number of the record, or 'subject_ID'

        Returns:
        int, the `patient_id` corr. to `rec_no`
        """
        raise NotImplementedError
    

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
        

    def load_data(self, rec_no:int) -> np.ndarray:
        """ finished, not checked,

        Parameters:
        -----------
        rec_no: int,
            number of the record, NOTE that rec_no starts from 1
        seg_no: int,
            number of the segment measured from the subject
        
        Returns:
        --------
        ndarray, the ppg data
        """
        rec_fp = os.path.join(self.db_path, "A{0:04d}".format(rec_no) + self.rec_ext)
        data = loadmat(rec_fp)
        data = np.asarray(data['val'], dtype=np.float64)
        
        if self.verbose >= 2:
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(figsize=(8,4))
            ax.plot(np.arange(0,len(data)/freq,1/freq),data)
            plt.show()
        
        return data


    def load_ann(self, rec_no:int) -> dict:
        """ finished, not checked,
        
        Parameters:
        -----------
        rec_no: int, optional,
            number of the record, or 'subject_ID',
            if not specified, then all annotations will be returned
        
        Returns:
        --------
        ann_dict, dict,
            the annotations with items:
            
        """
        assert rec_no in range(1, self.nb_records+1), "rec_no should be in range(1,{})".format(self.nb_records+1)
        ann_fp = os.path.join(self.db_path, "A{0:04d}".format(rec_no) + self.ann_ext)
        with open(input_header_file,'r') as f:
            header_data = f.readlines()
        header_data = [l.replace("\n", "") for l in header_data]

        ann_dict = {}
        ann_dict['rec_name'], ann_dict['nb_leads'], ann_dict['freq'], ann_dict['nb_samples'], ann_dict['datetime'], daytime = header_data[0].split(' ')
        ann_dict['nb_leads'] = int(ann_dict['nb_leads'])
        ann_dict['freq'] = int(ann_dict['freq'])
        ann_dict['nb_samples'] = int(ann_dict['nb_samples'])
        ann_dict['datetime'] = datetime.strptime(' '.join([ann_dict['datetime'], daytime]), '%d-%b-%Y %H:%M:%S')
        ann_dict['age'] = int([l for l in header_data if l.startswith('#Age')][0].split(": ")[-1])
        ann_dict['sex'] = [l for l in header_data if l.startswith('#Sex')][0].split(": ")[-1]
        ann_dict['diagnosis'] = [l for l in header_data if l.startswith('#Dx')][0].split(": ")[-1].split(",")
        for idx, d in enumerate(ann_dict['diagnosis']):
            if d == 'Normal':
                ann_dict['diagnosis'] = 'N'
        ann_dict['medical_prescription'] = [l for l in header_data if l.startswith('#Rx')][0].split(": ")[-1]
        ann_dict['history'] = [l for l in header_data if l.startswith('#Hx')][0].split(": ")[-1]
        ann_dict['symptom_or_surgery'] = [l for l in header_data if l.startswith('#Sx')][0].split(": ")[-1]
        df_leads = pd.read_csv(io.StringIO('\n'.join(header_data[1:13])), delim_whitespace=True, header=None)
        df_leads.columns = ['filename', 'res+offset', 'resolution(mV)', 'ADC', 'baseline', 'first_value', 'checksum', 'redundant', 'lead_name']
        df_leads['resolution(bits)'] = df_leads['res+offset'].apply(lambda s: s.split('+')[0])
        df_leads['offset'] = df_leads['res+offset'].apply(lambda s: s.split('+')[1])
        df_leads = df_leads[['filename', 'resolution(bits)', 'offset', 'resolution(mV)', 'ADC', 'baseline', 'first_value', 'checksum', 'lead_name']]
        df_leads['resolution(mV)'] = df_leads['resolution(mV)'].apply(lambda s: s.split('/')[0])
        for k in ['resolution(bits)', 'offset', 'resolution(mV)', 'ADC', 'baseline', 'first_value', 'checksum']:
            df_leads[k] = df_leads[k].apply(lambda s: int(s))
        ann_dict['df_leads'] = df_leads

        return ann_dict


    def get_labels(self, rec_no:int) -> List[str]:
        """ finished, not checked,
        
        Parameters:
        -----------
        rec_no: int,
            number of the record
        
        Returns:
        --------
        labels, list,
            the list of labels (abbr. diagnosis)
        """
        ann_dict = self.load_ann(rec_no)
        labels = ann_dict['diagnosis']
        return labels


    def get_diagnosis(self, rec_no:int) -> List[str]:
        """ finished, not checked,
        
        Parameters:
        -----------
        rec_no: int,
            number of the record
        
        Returns:
        --------
        diagonosis, list,
            the list of (full) diagnosis
        """
        diagonosis = self.get_labels(rec_no)
        diagonosis = [self.diagnosis_abbr_to_full(item) for item in diagonosis]
        return diagonosis


    def get_patient_info(self, rec_no:int, items:Optional[List[str]]=None, verbose:int=2) -> dict:
        """ finished, not checked,

        Parameters:
        -----------
        rec_no: int,
            number of the record, or 'subject_ID'
        items: list of str, optional,
            items of the patient information (e.g. sex, age, etc.)
        
        Returns:
        --------
        patient_info, dict,
        """
        if items is None or len(items) == 0:
            info_items = [
                'age', 'sex', 'medical_prescription', 'history', 'symptom_or_surgery',
            ]
        else:
            info_items = items
        ann_dict = self.load_ann(rec_no)
        patient_info = [ann_dict[item] for item in info_items]

        return patient_info


    def plot(self, leads:Optional[List[str]]=None, **kwargs) -> NoReturn:
        """
        """
        raise NotImplementedError
