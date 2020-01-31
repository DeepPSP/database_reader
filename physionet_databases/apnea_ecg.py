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
    "ApneaECG",
]


class ApneaECG(PhysioNetDataBase):
    """ Finished, to be improved,

    Apnea-ECG Database

    About apnea-ecg (CinC 2000):
    --------------------------------------
    1. consist of 70 records, divided into a learning set of 35 records (a01 through a20, b01 through b05, and c01 through c10), and a test set of 35 records (x01 through x35)
    2. recordings vary in length from slightly less than 7 hours to nearly 10 hours (401 - 578 min) each
    3. control group (c01 through c10): records having fewer than 5 min of disorder breathing
    4. borderline group (b01 through b05): records having 10-96 min of disorder breathing
    5. apnea group (a01 through a20): records having 100 min or more of disorder breathing
    6. .dat files contain the digitized ECGs
    7. .apn files are (binary) annotation files (only for the learning set), containing an annotation for each minute of each recording indicating the presence or absence of apnea at that time. labels are in the member 'symbol', 'N' for normal, 'A' for apnea
    8. .qrs files are machine-generated (binary) annotation files, unaudited and containing errors, provided for the convenience of those who do not wish to use their own QRS detectors
    9. c05 and c06 come from the same original recording (c05 begins 80 seconds later than c06). c06 may have been a corrected version of c05
    10. eight records (a01 through a04, b01, and c01 through c03) that include respiration signals have several additional files each.
    11. *r.* files contains respiration information correspondingly

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. sleep apnea

    References:
    -----------
    [1] https://physionet.org/content/apnea-ecg/1.0.0/
    [2] T Penzel, GB Moody, RG Mark, AL Goldberger, JH Peter. The Apnea-ECG Database. Computers in Cardiology 2000;27:255-258
    """
    def __init__(self, db_path:Optional[str]=None, **kwargs):
        super().__init__(db_name='apnea-ecg', db_path=db_path, **kwargs)
        self.freq = 100
        try:
            self.all_records = wfdb.get_record_list('apnea-ecg')
        except:
            try:
                self.all_records = os.listdir(self.db_path)
                self.all_records = list(set([item.split('.')[0] for item in self.all_records]))
            except:
                self.all_records = ['a01', 'a01er', 'a01r', 'a02', 'a02er', 'a02r', 'a03', 'a03er', 'a03r', 'a04', 'a04er', 'a04r', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'b01', 'b01er', 'b01r', 'b02', 'b03', 'b04', 'b05', 'c01', 'c01er', 'c01r', 'c02', 'c02er', 'c02r', 'c03', 'c03er', 'c03r', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10', 'x01', 'x02', 'x03', 'x04', 'x05', 'x06', 'x07', 'x08', 'x09', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35']
        self.learning_set = [r for r in self.all_records if 'x' not in r and 'r' not in r]
        self.test_set = [r for r in self.all_records if 'x' in r and 'r' not in r]
        self.control_group = [r for r in self.learning_set if 'c' in r]
        self.borderline_group = [r for r in self.learning_set if 'b' in r]
        self.apnea_group = [r for r in self.learning_set if 'a' in r]

        self.sleep_event_keys = ['event_name', 'event_start', 'event_end', 'event_duration']
        self.palette = {
            'Obstructive Apnea': 'yellow',
        }


    def get_subject_id(self, rec:str) -> int:
        """

        Parameters:
        -----------
        rec: str,
            record name
        """
        stoi = {'a':'1', 'b':'2', 'c':'3', 'x':'4'}
        return int('2000' + stoi[rec[0]] + rec[1:])


    def database_info(self, detailed:bool=False) -> NoReturn:
        """

        Parameters:
        -----------
        detailed: bool, default False,
            if False, physionet's "db_description" will be printed,
            if True, then docstring of the class will be printed additionally
        """
        short_description = self.df_all_db_info[self.df_all_db_info['db_name']==self.db_name]['db_description'].values[0]
        print(short_description)

        if detailed:
            print(self.__doc__)


    def load_ecg_data(self, rec:str, lead:int=0, rec_path:Optional[str]=None) -> np.ndarray:
        """

        Parameters:
        -----------
        rec: str,
            record name
        lead: int, default 0
            number of the lead, can be 0 or 1
        rec_path: str, optional,
            path of the file which contains the ecg data,
            if not given, default path will be used
        """
        file_path = rec_path if rec_path is not None else self.db_path+rec
        return wfdb.rdrecord(file_path).p_signal[:,0]


    def load_ann(self, rec:str, ann_path:Optional[str]=None) -> list:
        """

        Parameters:
        -----------
        rec: str,
            record name
        ann_path: str, optional,
            path of the file which contains the annotations,
            if not given, default path will be used
        """
        file_path = ann_path if ann_path is not None else self.db_path+rec
        anno = wfdb.rdann(file_path, extension='apn')
        detailed_anno = [[si//(self.freq*60), sy] for si, sy in zip(anno.sample, anno.symbol)]
        return detailed_anno


    def load_apnea_event_ann(self, rec:str, ann_path:Optional[str]=None) -> pd.DataFrame:
        """

        Parameters:
        -----------
        rec: str,
            record name
        ann_path: str, optional,
            path of the file which contains the annotations,
            if not given, default path will be used
        """
        detailed_anno = self.load_ann(rec, ann_path)
        apnea = np.array([p[0] for p in detailed_anno if p[1] == 'A'])

        if len(apnea) > 0:
            apnea_endpoints = [apnea[0]]
            # TODO: check if split_indices is correctly computed
            split_indices = np.where(np.diff(apnea)>1)[0].tolist()
            for i in split_indices:
                apnea_endpoints += [apnea[i], apnea[i+1]]
            apnea_endpoints.append(apnea[-1])

            apnea_periods = []
            for i in range(len(apnea_endpoints)//2):
                pe = [apnea_endpoints[2*i], apnea_endpoints[2*i+1]]
                apnea_periods.append(pe)
        else:
            apnea_periods = []
        
        if self.verbose >= 1:
            if len(apnea_periods) > 0:
                print('apnea period(s) (units in minutes) of record {} is(are): {}'.format(rec, apnea_periods))
            else:
                print('record {} has no apnea period'.format(rec))

        if len(apnea_periods) == 0:
            return pd.DataFrame(columns=self.sleep_event_keys)

        apnea_periods = np.array([[60*p[0], 60*p[1]] for p in apnea_periods])  # minutes to seconds
        apnea_periods = np.array(apnea_periods,dtype=int).reshape((len(apnea_periods),2))

        df_apnea_ann = pd.DataFrame(apnea_periods,columns=['event_start','event_end'])
        df_apnea_ann['event_name'] = 'Obstructive Apnea'
        df_apnea_ann['event_duration'] = df_apnea_ann.apply(lambda row: row['event_end']-row['event_start'], axis=1)

        df_apnea_ann = df_apnea_ann[self.sleep_event_keys]

        return df_apnea_ann


    def plot_ann(self, rec, ann_path:Optional[str]=None) -> NoReturn:
        """

        Parameters:
        -----------
        rec: str,
            record name
        ann_path: str, optional,
            path of the file which contains the annotations,
            if not given, default path will be used
        """
        df_apnea_ann = self.load_apnea_event_ann(rec, ann_path)
        self._plot_ann(df_apnea_ann)


    def _plot_ann(self, df_apnea_ann:pd.DataFrame) -> NoReturn:
        """

        Parameters:
        df_apnea_ann: DataFrame,
            apnea events with columns `self.sleep_event_keys`
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        patches = {k: mpatches.Patch(color=c, label=k) for k,c in self.palette.items()}
        _, ax = plt.subplots(figsize=(20,4))
        plot_alpha = 0.5
        for _, row in df_apnea_ann.iterrows():
            ax.axvspan(datetime.fromtimestamp(row['event_start']), datetime.fromtimestamp(row['event_end']), color=self.palette[row['event_name']], alpha=plot_alpha)
            ax.legend(handles=[patches[k] for k in self.palette.keys()],loc='best')  # keep ordering
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='y', which='both', length=0)
