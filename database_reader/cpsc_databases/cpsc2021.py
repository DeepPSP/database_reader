# -*- coding: utf-8 -*-
"""
"""
import os
import random
import math
import time
import warnings
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, Dict, Sequence, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
from scipy.signal import resample, resample_poly
from easydict import EasyDict as ED
import wfdb

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
    get_record_list_recursive3,
    ms2samples, samples2ms,
    DEFAULT_FIG_SIZE_PER_SEC,
)
from ..utils.utils_universal import generalized_intervals_intersection
from ..base import OtherDataBase


__all__ = [
    "CPSC2021",
]


# configurations for visualization
PlotCfg = ED()
# default const for the plot function in dataset.py
# used only when corr. values are absent
# all values are time bias w.r.t. corr. peaks, with units in ms
PlotCfg.p_onset = -40
PlotCfg.p_offset = 40
PlotCfg.q_onset = -20
PlotCfg.s_offset = 40
PlotCfg.qrs_radius = 60
PlotCfg.t_onset = -100
PlotCfg.t_offset = 60


class CPSC2021(OtherDataBase):
    r"""

    The 4th China Physiological Signal Challenge 2021:
    Paroxysmal Atrial Fibrillation Events Detection from Dynamic ECG Recordings

    ABOUT CPSC2021:
    ---------------
    1. source ECG data are recorded from 12-lead Holter or 3-lead wearable ECG monitoring devices
    2. dataset provides variable-length ECG fragments extracted from lead I and lead II of the long-term source ECG data, each sampled at 200 Hz
    3. AF event is limited to be no less than 5 heart beats
    4. training set in the 1st stage consists of 716 records, extracted from the Holter records from 12 AF patients and 42 non-AF patients (usually including other abnormal and normal rhythms)
    5. test set comprises data from the same source as the training set as well as DIFFERENT data source, which are NOT to be released at any point
    6. annotations are standardized according to PhysioBank Annotations (Ref. [2] or PhysioNetDataBase.helper), and include the beat annotations (R peak location and beat type), the rhythm annotations (rhythm change flag and rhythm type) and the diagnosis of the global rhythm
    7. classification of a record is stored in corresponding .hea file, which can be accessed via the attribute `comments` of a wfdb Record obtained using `wfdb.rdheader`, `wfdb.rdrecord`, and `wfdb.rdsamp`; beat annotations and rhythm annotations can be accessed using the attributes `symbol`, `aux_note` of a wfdb Annotation obtained using `wfdb.rdann`, corresponding indices in the signal can be accessed via the attribute `sample`
    8. challenge task:
        (1). clasification of rhythm types: non-AF rhythm (N), persistent AF rhythm (AFf) and paroxysmal AF rhythm (AFp)
        (2). locating of the onset and offset for any AF episode prediction
    9. challenge metrics:
        (1) metrics (Ur, scoring matrix) for classification:
                Prediction
                N        AFf        AFp
        N      +1        -1         -0.5
        AFf    -2        +1          0
        AFp    -1         0         +1
        (2) metric (Ue) for detecting onsets and offsets for AF events (episodes):
        +1 if the detected onset (or offset) is within ±1 beat of the annotated position, and +0.5 if within ±2 beats
        (3) final score (U):
        U = \dfrac{1}{N} \sum\limits_{i=1}^N \left( Ur_i + \dfrac{Ma_i}{\max\{Mr_i, Ma_i\}} \right)
        where N is the number of records, Ma is the number of annotated AF episodes, Mr the number of predicted AF episodes

    NOTE:
    -----
    1. if an ECG record is classified as AFf, the provided onset and offset locations should be the first and last record points. If an ECG record is classified as N, the answer should be an empty list
    2. it can be inferred from the classification scoring matrix that the punishment of false negatives of AFf is very heavy, while mixing-up of AFf and AFp is not punished

    ISSUES:
    -------
    1. 

    TODO:
    -----
    1. 

    Usage:
    ------
    1. AF (event, fine) detection

    References:
    -----------
    [1] http://www.icbeb.org/CPSC2021
    [2] https://archive.physionet.org/physiobank/annotations.shtml
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """ finished, checked,

        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="CPSC2021", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)

        self.fs = 200
        self.spacing = 1000/self.fs
        self.rec_ext = "dat"
        self.ann_ext = "atr"
        self.all_leads = ["I", "II"]

        self._labels_f2a = { # fullname to abbreviation
            "non atrial fibrillation": "N",
            "paroxysmal atrial fibrillation": "AFp",
            "persistent atrial fibrillation": "AFf",
        }
        self._labels_f2n = { # fullname to number
            "non atrial fibrillation": 0,
            "paroxysmal atrial fibrillation": 1,
            "persistent atrial fibrillation": 2,
        }

        self._ls_rec()

        self._epsilon = 1e-7  # dealing with round(0.5) = 0, hence keeping accordance with output length of `resample_poly`

        # self.palette = {"spb": "yellow", "pvc": "red",}


    def _ls_rec(self) -> NoReturn:
        """ finished, checked,

        list all the records and load into `self._all_records`,
        facilitating further uses
        """
        fn = "RECORDS"
        record_list_fp = os.path.join(self.db_dir, fn)
        if os.path.isfile(record_list_fp):
            with open(record_list_fp, "r") as f:
                self._all_records = f.read().splitlines()
        else:
            print("Please wait patiently to let the reader find all records...")
            start = time.time()
            rec_patterns_with_ext = f"data_(?:\d+)_(?:\d+).{self.rec_ext}"
            self._all_records = \
                get_record_list_recursive3(self.db_dir, rec_patterns_with_ext)
            print(f"Done in {time.time() - start:.5f} seconds!")
            with open(record_list_fp, "w") as f:
                f.write("\n".join(self._all_records))


    def get_subject_id(self, rec:str) -> int:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        sid: int,
            subject id corresponding to the record
        """
        sid = int(rec.split("_")[1])
        return sid


    def load_data(self, rec:str, leads:Optional[Union[str, List[str]]]=None, data_format:str="channel_first", units:str="mV", fs:Optional[Real]=None) -> np.ndarray:
        """ finished, checked,

        load physical (converted from digital) ecg data,
        which is more understandable for humans

        Parameters:
        -----------
        rec: str,
            name of the record
        leads: str or list of str, optional,
            the leads to load
        data_format: str, default "channel_first",
            format of the ecg data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        fs: real number, optional,
            if not None, the loaded data will be resampled to this sampling frequency
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        assert data_format.lower() in ["channel_first", "lead_first", "channel_last", "lead_last"]
        if not leads:
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads
        assert all([l in self.all_leads for l in _leads])

        rec_fp = os.path.join(self.db_dir, rec)
        wfdb_rec = wfdb.rdrecord(rec_fp, physical=True, channel_names=_leads)
        data = np.asarray(wfdb_rec.p_signal.T)
        # lead_units = np.vectorize(lambda s: s.lower())(wfdb_rec.units)

        if units.lower() in ["uv", "μv"]:
            data = data * 1000

        if fs is not None and fs != self.fs:
            data = resample_poly(data, fs, self.fs, axis=1)

        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T

        return data

    
    def load_ann(self, rec:str, field:Optional[str]=None, **kwargs:Any) -> Union[dict, np.ndarray, List[List[int]], str]:
        """ finished, checked,

        load annotations of the record

        Parameters:
        -----------
        rec: str,
            name of the record
        field: str, optional
            field of the annotation, can be one of "rpeaks", "af_episodes", "label",
            if not specified, all fields of the annotation will be returned in the form of a dict
        kwargs: dict,
            key word arguments for functions loading rpeaks, af_episodes, and label respectively,
            including:
            fs: int, optional,
                the resampling frequency
            fmt: str,
                format of af_episodes, or format of label,
                for more details, ref. corresponding functions
            used only when field is specified,

        Returns:
        --------
        ann: dict, or list, or ndarray, or str,
            annotaton of the record
        """
        func = {
            "rpeaks": self.load_rpeaks,
            "af_episodes": self.load_af_episodes,
            "label": self.load_label,
        }
        if field is None:
            ann = {k: f(rec) for k,f in func.items()}
            if kwargs:
                warnings.warn(f"key word arguments {list(kwargs.keys())} ignored when field is not specified!")
            return ann

        try:
            f = func[field.lower()]
        except:
            raise ValueError(f"invalid field")
        ann = f(rec, **kwargs)
        return ann


    def load_rpeaks(self, rec:str, fs:Optional[Real]=None) -> np.ndarray:
        """ finished, checked,

        load position (in terms of samples) of rpeaks

        Parameters:
        -----------
        rec: str,
            name of the record
        fs: real number, optional,
            if not None, positions of the loaded rpeaks will be ajusted according to this sampling frequency

        Returns:
        --------
        rpeaks: ndarray,
            position (in terms of samples) of rpeaks of the record
        """
        rpeaks = wfdb.rdann(os.path.join(self.db_dir, rec), extension=self.ann_ext).sample
        if fs is not None and fs!=self.fs:
            rpeaks = np.round(rpeaks * fs / self.fs + self._epsilon).astype(int)
        return rpeaks


    def load_af_episodes(self, rec:str, fs:Optional[Real]=None, fmt:str="intervals") -> Union[List[List[int]], np.ndarray]:
        """ finished, checked,

        load the episodes of atrial fibrillation, in terms of intervals or mask

        Paramters:
        ----------
        rec: str,
            name of the record
        fs: real number, optional,
            if not None, positions of the loaded intervals or mask will be ajusted according to this sampling frequency
        fmt: str, default "intervals",
            format of the episodes of atrial fibrillation, can be one of "intervals", "mask"

        Returns:
        --------
        af_episodes: list or ndarray,
            episodes of atrial fibrillation, in terms of intervals or mask
        """
        header = wfdb.rdheader(os.path.join(self.db_dir, rec))
        label = self._labels_f2a[header.comments[0]]
        siglen = header.sig_len
        ann = wfdb.rdann(os.path.join(self.db_dir, rec), extension=self.ann_ext)
        aux_note = np.array(ann.aux_note)
        rpeaks = ann.sample
        af_start_inds = np.where((aux_note=="(AFIB") | (aux_note=="(AFL"))[0]
        af_end_inds = np.where(aux_note=="(N")[0]
        assert len(af_start_inds) == len(af_end_inds), "unequal number of af period start indices and af period end indices"
        intervals = []
        for start, end in zip(af_start_inds, af_end_inds):
            itv = [rpeaks[start], rpeaks[end]]
            intervals.append(itv)
        if fs is not None and fs != self.fs:
            if label == "AFf":
                # ref. NOTE. 1 of the class docstring
                # the `ann.sample` does not always satify this point after resampling
                intervals = [[0, self._round(siglen*fs/self.fs)-1]]
            else:
                intervals = [[self._round(itv[0]*fs/self.fs), self._round(itv[1]*fs/self.fs)] for itv in intervals]
        af_episodes = intervals

        if fmt.lower() in ["mask",]:
            if fs is not None and fs != self.fs:
                siglen = self._round(siglen*fs/self.fs)
            mask = np.zeros((siglen,), dtype=int)
            for itv in intervals:
                mask[itv[0]:itv[1]] = 1
            af_episodes = mask

        return af_episodes


    def load_label(self, rec:str, fmt:str="a") -> str:
        """ finished, checked,

        load (classifying) label of the record,
        among the following three classes:
        "non atrial fibrillation",
        "paroxysmal atrial fibrillation",
        "persistent atrial fibrillation",

        Parameters:
        -----------
        rec: str,
            name of the record
        fmt: str, default "a",
            format of the label, case in-sensitive, can be one of:
            "f", "fullname": the full name of the label
            "a", "abbr", "abbrevation": abbreviation for the label
            "n", "num", "number": class number of the label (in accordance with the settings of the offical class map)

        Returns:
        --------
        label: str,
            classifying label of the record
        """
        header = wfdb.rdheader(os.path.join(self.db_dir, rec))
        label = header.comments[0]
        if fmt.lower() in ["a", "abbr", "abbreviation"]:
            label = self._labels_f2a[label]
        elif fmt.lower() in ["n", "num", "number"]:
            label = self._labels_f2n[label]
        elif not fmt.lower() in ["f", "fullname"]:
            raise ValueError(f"format `{fmt}` of labels is not supported!")
        return label


    def plot(self, rec:str, data:Optional[np.ndarray]=None, ann:Optional[Dict[str, np.ndarray]]=None, ticks_granularity:int=0, leads:Optional[Union[str, List[str]]]=None, waves:Optional[Dict[str, Sequence[int]]]=None, **kwargs) -> NoReturn:
        """ finished, checked, to improve,

        plot the signals of a record or external signals (units in μV),
        with metadata (labels, episodes of atrial fibrillation, etc.),
        possibly also along with wave delineations

        Parameters:
        -----------
        rec: str,
            name of the record
        data: ndarray, optional,
            (2-lead) ecg signal to plot,
            should be of the format "channel_first", and compatible with `leads`
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ann: dict, optional,
            annotations for `data`,
            ignored if `data` is None
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads: str or list of str, optional,
            the leads to plot
        waves: dict, optional,
            indices of the wave critical points, including
            "p_onsets", "p_peaks", "p_offsets",
            "q_onsets", "q_peaks", "r_peaks", "s_peaks", "s_offsets",
            "t_onsets", "t_peaks", "t_offsets"
        kwargs: dict,

        TODO:
        -----
        1. slice too long records, and plot separately for each segment
        2. plot waves using `axvspan`

        NOTE:
        -----
        1. `Locator` of `plt` has default `MAXTICKS` equal to 1000,
        if not modifying this number, at most 40 seconds of signal could be plotted once
        2. raw data usually have very severe baseline drifts,
        hence the isoelectric line is not plotted

        Contributors: Jeethan, and WEN Hao
        """
        if "plt" not in dir():
            import matplotlib.pyplot as plt
            plt.MultipleLocator.MAXTICKS = 3000
        if leads is None or leads == "all":
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads
        assert all([l in self.all_leads for l in _leads])

        if data is None:
            _data = self.load_data(rec, leads=_leads, data_format="channel_first", units="μV")
        else:
            units = self._auto_infer_units(data)
            print(f"input data is auto detected to have units in {units}")
            if units.lower() == "mv":
                _data = 1000 * data
            else:
                _data = data
            assert _data.shape[0] == len(_leads), \
                f"number of leads from data of shape ({_data.shape[0]}) does not match the length ({len(_leads)}) of `leads`"

        if waves:
            if waves.get("p_onsets", None) and waves.get("p_offsets", None):
                p_waves = [
                    [onset, offset] \
                        for onset, offset in zip(waves["p_onsets"], waves["p_offsets"])
                ]
            elif waves.get("p_peaks", None):
                p_waves = [
                    [
                        max(0, p + ms2samples(PlotCfg.p_onset, fs=self.get_fs(rec))),
                        min(_data.shape[1], p + ms2samples(PlotCfg.p_offset, fs=self.get_fs(rec)))
                    ] for p in waves["p_peaks"]
                ]
            else:
                p_waves = []
            if waves.get("q_onsets", None) and waves.get("s_offsets", None):
                qrs = [
                    [onset, offset] for onset, offset in zip(waves["q_onsets"], waves["s_offsets"])
                ]
            elif waves.get("q_peaks", None) and waves.get("s_peaks", None):
                qrs = [
                    [
                        max(0, q + ms2samples(PlotCfg.q_onset, fs=self.get_fs(rec))),
                        min(_data.shape[1], s + ms2samples(PlotCfg.s_offset, fs=self.get_fs(rec)))
                    ] for q,s in zip(waves["q_peaks"], waves["s_peaks"])
                ]
            elif waves.get("r_peaks", None):
                qrs = [
                    [
                        max(0, r + ms2samples(PlotCfg.qrs_radius, fs=self.get_fs(rec))),
                        min(_data.shape[1], r + ms2samples(PlotCfg.qrs_radius, fs=self.get_fs(rec)))
                    ] for r in waves["r_peaks"]
                ]
            else:
                qrs = []
            if waves.get("t_onsets", None) and waves.get("t_offsets", None):
                t_waves = [
                    [onset, offset] for onset, offset in zip(waves["t_onsets"], waves["t_offsets"])
                ]
            elif waves.get("t_peaks", None):
                t_waves = [
                    [
                        max(0, t + ms2samples(PlotCfg.t_onset, fs=self.get_fs(rec))),
                        min(_data.shape[1], t + ms2samples(PlotCfg.t_offset, fs=self.get_fs(rec)))
                    ] for t in waves["t_peaks"]
                ]
            else:
                t_waves = []
        else:
            p_waves, qrs, t_waves = [], [], []
        palette = {"p_waves": "green", "qrs": "yellow", "t_waves": "pink",}
        plot_alpha = 0.4

        if ann is None or data is None:
            _ann = self.load_ann(rec)
            rpeaks = _ann["rpeaks"]
            af_episodes = _ann["af_episodes"]
            label = _ann["label"]
        else:
            rpeaks = ann.get("rpeaks", [])
            af_episodes = ann.get("af_episodes", [])
            label = ann["label"]

        nb_leads = len(_leads)

        line_len = self.fs * 25  # 25 seconds
        nb_lines = math.ceil(_data.shape[1]/line_len)

        bias_thr = 0.1
        winL = 0.06
        winR = 0.08

        for idx in range(nb_lines):
            seg = _data[..., idx*line_len: (idx+1)*line_len]
            secs = (np.arange(seg.shape[1]) + idx*line_len) / self.fs
            fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * seg.shape[1] / self.fs))
            # if same_range:
            #     y_ranges = np.ones((seg.shape[0],)) * np.max(np.abs(seg)) + 100
            # else:
            #     y_ranges = np.max(np.abs(seg), axis=1) + 100
            # fig_sz_h = 6 * y_ranges / 1500
            fig_sz_h = 6 * sum([seg_lead.max() - seg_lead.min() + 200 for seg_lead in seg]) / 1500
            fig, axes = plt.subplots(nb_leads, 1, sharex=True, figsize=(fig_sz_w, np.sum(fig_sz_h)))
            if nb_leads == 1:
                axes = [axes]
        
            for ax_idx in range(nb_leads):
                axes[ax_idx].plot(secs, seg[ax_idx], color="black", label=f"lead - {_leads[ax_idx]}")
                # axes[ax_idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
                # NOTE that `Locator` has default `MAXTICKS` equal to 1000
                if ticks_granularity >= 1:
                    axes[ax_idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                    axes[ax_idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                    axes[ax_idx].grid(which="major", linestyle="-", linewidth="0.5", color="red")
                if ticks_granularity >= 2:
                    axes[ax_idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                    axes[ax_idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                    axes[ax_idx].grid(which="minor", linestyle=":", linewidth="0.5", color="black")
                # add extra info. to legend
                # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
                axes[ax_idx].plot([], [], " ", label=f"label - {label}")
                seg_rpeaks = [r/self.fs for r in rpeaks if idx*line_len <= r < (idx+1)*line_len]
                for r in seg_rpeaks:
                    axes[ax_idx].axvspan(
                    max(secs[0], r-bias_thr), min(secs[-1], r+bias_thr),
                    color=palette["qrs"], alpha=0.3
                )
                seg_af_episodes = generalized_intervals_intersection(
                    af_episodes,
                    [[idx*line_len, (idx+1)*line_len]],
                )
                seg_af_episodes = [[itv[0]-idx*line_len, itv[1]-idx*line_len] for itv in seg_af_episodes]
                for itv_start, itv_end in seg_af_episodes:
                    axes[ax_idx].plot(secs[itv_start:itv_end], seg[ax_idx,itv_start:itv_end], color="red")
                for w in ["p_waves", "qrs", "t_waves"]:
                    for itv in eval(w):
                        axes[ax_idx].axvspan(itv[0], itv[1], color=palette[w], alpha=plot_alpha)
                axes[ax_idx].legend(loc="upper left")
                axes[ax_idx].set_xlim(secs[0], secs[-1])
                # axes[ax_idx].set_ylim(-y_ranges[ax_idx], y_ranges[ax_idx])
                axes[ax_idx].set_xlabel("Time [s]")
                axes[ax_idx].set_ylabel("Voltage [μV]")
            plt.subplots_adjust(hspace=0.2)
        plt.show()


    def _round(self, n:Real) -> int:
        """ finished, checked,

        dealing with round(0.5) = 0, hence keeping accordance with output length of `resample_poly`
        """
        return int(round(n + self._epsilon))