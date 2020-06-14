# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy import interpolate
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, NoReturn
from numbers import Real

from ..utils.common import ArrayLike
from ..utils.utils_signal import resample_irregular_timeseries
from ..base import OtherDataBase


__all__ = [
    "SleepAccel",
]

class SleepAccel(OtherDataBase):
    """

    ABOUT sleep_accel:
    ------------------
    1. sleep stages in the records:
        wake = 0
        n1 = 1
        n2 = 2
        n3 = 3
        n4 = 4
        rem = 5
        unscored = -1
    2. 

    NOTE:
    -----

    ISSUES:
    -------
    1. 

    Usage:
    ------
    1. sleep stage

    References:
    -----------
    [1] Walch O, Huang Y, Forger D, et al. Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device[J]. Sleep, 2019, 42(12): zsz180.
    [2] https://github.com/ojwalch/sleep_classifiers/
    [3] https://alpha.physionet.org/content/sleep-accel/1.0.0/
    [4] to add acc_to_count references
    """
    def __init__(self, db_path:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """ not finished,

        Parameters:
        -----------
        db_path: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,

        typical 'db_path':  "/export/servers/data/sleep_accel/"
        """
        super().__init__(db_name="SleepAccel", db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)

        self.hr_path = os.path.join(db_path, "heart_rate")
        self.lb_path = os.path.join(db_path, "labels")
        self.motion_path = os.path.join(db_path, "motion")
        self.steps_path = os.path.join(db_path, "steps")

        self.hr_file_suffix = "_heartrate.txt"
        self.lb_file_suffix = "_labeled_sleep.txt"
        self.motion_file_suffix = "_acceleration.txt"
        self.steps_file_suffix = "_steps.txt"

        self.acc_freq = 50
        self.hr_freq = None  # to be checked
        self.all_subjects = ['1066528', '1360686', '1449548', '1455390', '1818471', '2598705', '2638030', '3509524', '3997827', '4018081', '4314139', '4426783', '46343', '5132496', '5383425', '5498603', '5797046', '6220552', '759667', '7749105', '781756', '8000685', '8173033', '8258170', '844359', '8530312', '8686948', '8692923', '9106476', '9618981', '9961348']
        # self.all_subjects = [item.split("_")[0] for item in os.listdir(self.motion_path)]

        self.to_conventional_lables = {
            -1: 0,  # unscored
            0: 6,  # wake
            5: 5,  # REM
            4: 1,  # N4
            3: 2,  # N3
            2: 3,  # N2
            1: 4,  # N1
        }
        self.to_binary_labels = {
            -1: 0,  # unscored
            0: 2,  # wake
            5: 1,  # sleep
            4: 1,  # sleep
            3: 1,  # sleep
            2: 1,  # sleep
            1: 1,  # sleep
        }


    def load_labels(self, subject_id:str) -> pd.DataFrame:
        """ finished, checked,

        load labels of sleep stages

        Paramters:
        ----------
        subject_id: str,
            subject id in `self.all_subjects`
        
        Returns:
        --------
        df_lb: DataFrame, with columns 'sec','sleep_stage'
        """
        fp = os.path.join(self.lb_path, subject_id+self.lb_file_suffix)
        df_lb = pd.read_csv(fp,sep=' ',header=None,names=['sec','sleep_stage'])
        df_lb['sleep_stage'] = df_lb['sleep_stage'].apply(lambda ss: self.to_conventional_lables[ss])
        return df_lb


    def load_motion_data(self, subject_id:str) -> pd.DataFrame:
        """ finished, checked,

        load motion (accelerometer) data

        Paramters:
        ----------
        subject_id: str,
            subject id in `self.all_subjects`
        
        Returns:
        --------
        df_mt: DataFrame, with columns 'sec','x','y','z'
        """
        mt_fp = os.path.join(self.motion_path, subject_id+self.motion_file_suffix)
        df_mt = pd.read_csv(mt_fp,sep=' ',header=None,names=['sec','x','y','z'])
        df_mt = df_mt.sort_values(by='sec').drop_duplicates(subset='sec').reset_index(drop=True)
        return df_mt


    def load_hr_data(self, subject_id:str) -> pd.DataFrame:
        """ finished, checked,

        load heart rate data

        Paramters:
        ----------
        subject_id: str,
            subject id in `self.all_subjects`
        
        Returns:
        --------
        df_hr: DataFrame, with columns 'sec','hr'
        """
        hr_fp = os.path.join(self.hr_path, subject_id+self.hr_file_suffix)
        df_hr = pd.read_csv(hr_fp,sep=',',header=None,names=['sec','hr'])
        df_hr = df_hr.sort_values(by='sec').drop_duplicates(subset='sec').reset_index(drop=True)
        return df_hr


    def load_step_data(self, subject_id:str) -> pd.DataFrame:
        """ finished, checked,

        load step count data

        Paramters:
        ----------
        subject_id: str,
            subject id in `self.all_subjects`
        
        Returns:
        --------
        df_sp: DataFrame, with columns 'sec','step_count'
        """
        sp_fp = os.path.join(self.steps_path, subject_id+self.steps_file_suffix)
        df_sp = pd.read_csv(sp_fp,sep=',',header=None,names=['sec','step_count'])
        df_sp = df_sp.sort_values(by='sec').drop_duplicates(subset='sec').reset_index(drop=True)
        return df_sp


    def plot_lb(self, subject_id:str, style:str, **kwargs) -> pd.DataFrame:
        """ finished, checked,

        plot labels of sleep stages

        Paramters:
        ----------
        subject_id: str,
            subject id in `self.all_subjects`
        style: str,
            to be implemented
        
        Returns:
        --------
        df_lb: DataFrame,
            ref. `self.load_labels`
        """
        import matplotlib.pyplot as plt
        df_lb = self.load_labels(subject_id)
        fig,ax = plt.subplots(figsize=(20,4))
        ax.plot(df_lb['sec'].values, df_lb['sleep_stage'].values)
        ax.set_yticks(np.arange(0,7,1))
        ax.set_yticklabels(['unscored','N4','N3','N2','N1','REM','wake'])
        plt.show()
        return df_lb


    def plot_mt_lb(self, subject_id:str, style:str, bin_state:bool=False, lag:Optional[Real]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ finished, checked,

        plot labels of sleep stages together with the triaxial motion data

        Paramters:
        ----------
        subject_id: str,
            subject id in `self.all_subjects`
        style: str,
            to be implemented
        bin_state: bool, default False,
            if True, labels of sleep stages will collapse into two states,
            namely sleep and wake (and unscored)
        lag: real number, optional,
            the time for `df_mt` to expand out from the time range of `df_lb`,
            if is None, time range of `df_mt` will be set the full time range of the motion data (which usually much wider)
        
        Returns:
        --------
        df_lb, df_mt: DataFrame,
            ref. `self.load_labels` and `self.load_motion_data`
        """
        import matplotlib.pyplot as plt
        df_lb = self.load_labels(subject_id)
        fig, ax_lb = plt.subplots(figsize=(20,4))
        if bin_state:
            df_lb['sleep_stage'] = df_lb['sleep_stage'].apply(lambda ss: self.to_binary_labels[ss])
            ax_lb.plot(df_lb['sec'].values, df_lb['sleep_stage'].values, color='red')
            ax_lb.set_yticks(np.arange(0,3,1))
            ax_lb.set_yticklabels(['unscored','sleep','wake'])
        else:
            df_lb['sleep_stage'] = df_lb['sleep_stage'].apply(lambda ss: self.to_conventional_labels[ss])
            ax_lb.plot(df_lb['sec'].values, df_lb['sleep_stage'].values, color='red')
            ax_lb.set_yticks(np.arange(0,7,1))
            ax_lb.set_yticklabels(['unscored','N4','N3','N2','N1','REM','wake'])
        
        lb_rg_t = df_lb.iloc[[0,-1]]['sec'].values
        df_mt = self.load_motion_data(subject_id)
        if lag is not None:
            lb_rg_t[0] = lb_rg_t[0]-lag
            lb_rg_t[-1] = lb_rg_t[-1]+lag
            df_mt = df_mt[(df_mt['sec']>=lb_rg_t[0])&(df_mt['sec']<=lb_rg_t[1])]
        ax_mt = ax_lb.twinx()
        ax_mt.plot(df_mt['sec'].values, df_mt['x'].values, label='x')
        ax_mt.plot(df_mt['sec'].values, df_mt['y'].values, label='y')
        ax_mt.plot(df_mt['sec'].values, df_mt['z'].values, label='z')
        ax_mt.legend(loc='best')
        return df_lb, df_mt


    def plot_ct_lb(self, subject_id:str, style:str, bin_state:bool=False, lag:Optional[Real]=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ not finished,

        plot labels of sleep stages together with the actigraph count data

        Paramters:
        ----------
        subject_id: str,
            subject id in `self.all_subjects`
        style: str,
            to be implemented
        bin_state: bool, default False,
            if True, labels of sleep stages will collapse into two states,
            namely sleep and wake (and unscored)
        lag: real number, optional,
            the time for `df_mt` to expand out from the time range of `df_lb`,
            if is None, time range of `df_mt` will be set the full time range of the motion data (which usually much wider)
        
        Returns:
        --------
        df_lb, df_ct, df_rsmpl, df_stats: DataFrame,
            ref. `self.load_labels` and `self.load_motion_data`
        """
        import matplotlib.pyplot as plt
        df_lb = self.load_labels(subject_id)
        fig, ax_lb = plt.subplots(figsize=(20,4))
        if bin_state:
            df_lb['sleep_stage'] = df_lb['sleep_stage'].apply(lambda ss: self.to_conventional_labels[ss])
            ax_lb.plot(df_lb['sec'].values, df_lb['sleep_stage'].values, color='red')
            ax_lb.set_yticks(np.arange(0,3,1))
            ax_lb.set_yticklabels(['unscored','sleep','wake'])
        else:
            df_lb['sleep_stage'] = df_lb['sleep_stage'].apply(lambda ss: self.to_binary_labels[ss])
            ax_lb.plot(df_lb['sec'].values, df_lb['sleep_stage'].values, color='red')
            ax_lb.set_yticks(np.arange(0,7,1))
            ax_lb.set_yticklabels(['unscored','N4','N3','N2','N1','REM','wake'])
        lb_rg_t = df_lb.iloc[[0,-1]]['sec'].values
        df_mt = self.load_motion_data(subject_id)
        if lag is not None:
            lb_rg_t[0] = lb_rg_t[0]-lag
            lb_rg_t[-1] = lb_rg_t[-1]+lag
            df_mt = df_mt[(df_mt['sec']>=lb_rg_t[0])&(df_mt['sec']<=lb_rg_t[1])]
        df_rsmpl = self.resample_motion_data(df_mt, output_fs=50)
        epoch_len = 60  # seconds
        ct_vals = self.acc_to_count(df_rsmpl, acc_fs=50, epoch_len=epoch_len)
        ct_secs = np.array([df_rsmpl.loc[0, 'sec']+idx*epoch_len for idx in range(len(ct_vals))])
        ct_secs_new = [-idx*epoch_len for idx in range(1,-ct_secs[0]//epoch_len+1) if -idx*epoch_len>=ct_secs[0]][::-1] + [idx*epoch_len for idx in range(ct_secs[-1]//epoch_len+1) if idx*epoch_len<ct_secs[-1]]
        f = interpolate.interp1d(ct_secs, ct_vals)
        ct_vals_new = f(ct_secs_new).astype(int)
        df_ct = pd.DataFrame()
        df_ct['sec'] = ct_secs_new
        df_ct['epoch_counts'] = ct_vals_new
        ax_ct = ax_lb.twinx()
        # ct_30s_val = [np.sum(np.linalg.norm(df_ct.loc[30*idx:30*(idx+1),['axis1','axis2','axis3']].values,axis=1)) for idx in range(len(df_ct)//30-1)]
        ax_ct.plot(ct_secs, ct_vals)
        # ax_ct.plot(df_ct['sec'].values, df_ct['axis1'].values, label='x')
        # ax_ct.plot(df_ct['sec'].values, df_ct['axis2'].values, label='y')
        # ax_ct.plot(df_ct['sec'].values, df_ct['axis3'].values, label='z')
        # ax_ct.legend(loc='best')
        plt.show()
        df_stats = df_ct.merge(df_lb, on='sec')
        self.logger.info("for subject {}, len(df_stats) = {}".format(subject_id, len(df_stats)))
        return df_lb, df_ct, df_rsmpl, df_stats


    def resample_motion_data(self, df_mt:pd.DataFrame, output_fs:Real) -> pd.DataFrame:
        """

        Parameters:
        -----------

        Returns:
        --------

        """
        acc_data = df_mt[['sec','x','y','z']].values
        acc_data[:,0] = np.vectorize(lambda t:round(1000*t))(acc_data[:,0])
        self.logger.info("acc_data.shape = {}".format(acc_data.shape))
        x_rsmpl = resample_irregular_timeseries(acc_data[:,[0,1]], output_fs=output_fs, return_with_time=True, method='interp1d', options={})
        y_rsmpl = resample_irregular_timeseries(acc_data[:,[0,2]], output_fs=output_fs, return_with_time=True, method='interp1d', options={})
        z_rsmpl = resample_irregular_timeseries(acc_data[:,[0,3]], output_fs=output_fs, return_with_time=True, method='interp1d', options={})
        df_rsmpl = pd.DataFrame()
        df_rsmpl['sec'] = x_rsmpl[:,0]/1000
        df_rsmpl['x'] = x_rsmpl[:,1]
        df_rsmpl['y'] = y_rsmpl[:,1]
        df_rsmpl['z'] = z_rsmpl[:,1]
        return df_rsmpl


    def acc_to_count(self, df_acc:pd.DataFrame, acc_fs:Real, epoch_len:Real, **kwargs) -> np.ndarray:
        """

        Parameters:
        -----------

        Returns:
        --------
        
        """
        raise NotImplementedError
