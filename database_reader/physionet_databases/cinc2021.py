# -*- coding: utf-8 -*-
"""
"""
import os, io, sys
import re
import json
import time
# import pprint
from copy import deepcopy
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Tuple, Set, Sequence, NoReturn
from numbers import Real, Number
from collections.abc import Iterable

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import wfdb
from scipy.io import loadmat
from scipy.signal import resample, resample_poly
from easydict import EasyDict as ED

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
    get_record_list_recursive3,
    ms2samples, samples2ms,
)
from ..utils.utils_signal import ensure_siglen
from ..utils.utils_misc import ecg_arrhythmia_knowledge as EAK
from ..utils.utils_misc.cinc2020_aux_data import (
    dx_mapping_all, dx_mapping_scored, dx_mapping_unscored,
    normalize_class, abbr_to_snomed_ct_code,
    df_weights_abbr,
    equiv_class_dict,
)
from ..utils.utils_universal.utils_str import dict_to_str
from ..utils.common import list_sum
from ..base import PhysioNetDataBase


__all__ = [
    "CINC2021",
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


class CINC2021(PhysioNetDataBase):
    """ NOT finished,

    Will Two Do? Varying Dimensions in Electrocardiography:
    The PhysioNet/Computing in Cardiology Challenge 2021

    ABOUT CINC2021:
    ---------------
    0. goal: build an algorithm that can classify cardiac abnormalities from either
        - twelve-lead (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)
        - six-lead (I, II, III, aVL, aVR, aVF),
        - four-lead (I, II, III, V2)
        - three-lead (I, II, V2)
        - two-lead (I, II)
    ECG recordings.
    1. tranches of data:
        - CPSC2018 (tranches A and B of CINC2020):
            contains 13,256 ECGs (6,877 from tranche A, 3,453 from tranche B),
            10,330 ECGs shared as training data, 1,463 retained as validation data,
            and 1,463 retained as test data.
            Each recording is between 6 and 144 seconds long with a sampling frequency of 500 Hz
        - INCARTDB (tranche C of CINC2020):
            contains 75 annotated ECGs,
            all shared as training data, extracted from 32 Holter monitor recordings.
            Each recording is 30 minutes long with a sampling frequency of 257 Hz
        - PTB (PTB and PTB-XL, tranches D and E of CINC2020):
            contains 22,353 ECGs,
            516 + 21,837, all shared as training data.
            Each recording is between 10 and 120 seconds long,
            with a sampling frequency of either 500 (PTB-XL) or 1,000 (PTB) Hz
        - Georgia (tranche F of CINC2020):
            contains 20,678 ECGs,
            10,334 ECGs shared as training data, 5,167 retained as validation data,
            and 5,167 retained as test data.
            Each recording is between 5 and 10 seconds long with a sampling frequency of 500 Hz
        - American (NEW, UNDISCLOSED):
            contains 10,000 ECGs,
            all retained as test data,
            geographically distinct from the Georgia database.
            Perhaps is the main part of the hidden test set of CINC2020
        - CUSPHNFH (NEW, the Chapman University, Shaoxing People’s Hospital and Ningbo First Hospital database)
            contains 45,152 ECGS,
            all shared as training data.
            Each recording is 10 seconds long with a sampling frequency of 500 Hz
            - Chapman_Shaoxing: "JS00001" - "JS10646"
            - Ningbo: "JS10647" - "JS45551"
    2. only a part of diagnosis_abbr (diseases that appear in the labels of the 6 tranches of training data) are used in the scoring function, while others are ignored. The scored diagnoses were chosen based on prevalence of the diagnoses in the training data, the severity of the diagnoses, and the ability to determine the diagnoses from ECG recordings. The ignored diagnosis_abbr can be put in a a "non-class" group.
    3. the (updated) scoring function has a scoring matrix with nonzero off-diagonal elements. This scoring function reflects the clinical reality that some misdiagnoses are more harmful than others and should be scored accordingly. Moreover, it reflects the fact that confusing some classes is much less harmful than confusing other classes.
    4. all data are recorded in the leads ordering of
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    using for example the following code:
    >>> db_dir = "/media/cfs/wenhao71/data/cinc2020_data/"
    >>> working_dir = "./working_dir"
    >>> dr = CINC2020Reader(db_dir=db_dir,working_dir=working_dir)
    >>> set_leads = []
    >>> for tranche, l_rec in dr.all_records.items():
    ...     for rec in l_rec:
    ...         ann = dr.load_ann(rec)
    ...         leads = ann["df_leads"]["lead_name"].values.tolist()
    ...     if leads not in set_leads:
    ...         set_leads.append(leads)

    NOTE:
    -----
    1. The datasets have been roughly processed to have a uniform format, hence differ from their original resource (e.g. differe in sampling frequency, sample duration, etc.)
    2. The original datasets might have richer metadata (especially those from PhysioNet), which can be fetched from corresponding reader's docstring or website of the original source
    3. Each sub-dataset might have its own organizing scheme of data, which should be carefully dealt with
    4. There are few "absolute" diagnoses in 12 lead ECGs, where large discrepancies in the interpretation of the ECG can be found even inspected by experts. There is inevitably something lost in translation, especially when you do not have the context. This doesn"t mean making an algorithm isn't important
    5. The labels are noisy, which one has to deal with in all real world data
    6. each line of the following classes are considered the same (in the scoring matrix):
        - RBBB, CRBBB (NOT including IRBBB)
        - PAC, SVPB
        - PVC, VPB
    7. unfortunately, the newly added tranches (C - F) have baseline drift and are much noisier. In contrast, CPSC data have had baseline removed and have higher SNR
    8. on Aug. 1, 2020, adc gain (including "resolution", "ADC"? in .hea files) of datasets INCART, PTB, and PTB-xl (tranches C, D, E) are corrected. After correction, (the .tar files of) the 3 datasets are all put in a "WFDB" subfolder. In order to keep the structures consistant, they are moved into "Training_StPetersburg", "Training_PTB", "WFDB" as previously. Using the following code, one can check the adc_gain and baselines of each tranche:
    >>> db_dir = "/media/cfs/wenhao71/data/CinC2021/"
    >>> working_dir = "./working_dir"
    >>> dr = CINC2021(db_dir=db_dir,working_dir=working_dir)
    >>> resolution = {tranche: set() for tranche in "ABCDEF"}
    >>> baseline = {tranche: set() for tranche in "ABCDEF"}
    >>> for tranche, l_rec in dr.all_records.items():
    ...     for rec in l_rec:
    ...         ann = dr.load_ann(rec)
    ...         resolution[tranche] = resolution[tranche].union(set(ann["df_leads"]["adc_gain"]))
    ...         baseline[tranche] = baseline[tranche].union(set(ann["df_leads"]["baseline"]))
    >>> print(resolution, baseline)
    {"A": {1000.0}, "B": {1000.0}, "C": {1000.0}, "D": {1000.0}, "E": {1000.0}, "F": {1000.0}} {"A": {0}, "B": {0}, "C": {0}, "D": {0}, "E": {0}, "F": {0}}
    9. the .mat files all contain digital signals, which has to be converted to physical values using adc gain, basesline, etc. in corresponding .hea files. `wfdb.rdrecord` has already done this conversion, hence greatly simplifies the data loading process.
    NOTE that there"s a difference when using `wfdb.rdrecord`: data from `loadmat` are in "channel_first" format, while `wfdb.rdrecord.p_signal` produces data in the "channel_last" format
    10. there"re 3 equivalent (2 classes are equivalent if the corr. value in the scoring matrix is 1):
        (RBBB, CRBBB), (PAC, SVPB), (PVC, VPB)
    11. in the newly (Feb., 2021) created dataset (ref. [7]), header files of each subset were gathered into one separate compressed file. This is due to the fact that updates on the dataset are almost always done in the header files. The correct usage of ref. [7], after uncompressing, is replacing the header files in the folder `All_training_WFDB` by header files from the 6 folders containing all header files from the 6 subsets. This procedure has to be done, since `All_training_WFDB` contains the very original headers with baselines: {"A": {1000.0}, "B": {1000.0}, "C": {1000.0}, "D": {2000000.0}, "E": {200.0}, "F": {4880.0}} (the last 3 are NOT correct)
    12. IMPORTANT: organization of the total dataset:
    either one moves all training records into ONE folder,
    or at least one moves the subsets Chapman_Shaoxing (WFDB_ChapmanShaoxing) and Ningbo (WFDB_Ningbo) into ONE folder, or use the data WFDB_ShaoxingUniv which is the union of WFDB_ChapmanShaoxing and WFDB_Ningbo

    Usage:
    ------
    1. ECG arrhythmia detection

    ISSUES: (all in CinC2020, left unfixed)
    -------
    1. reading the .hea files, baselines of all records are 0, however it is not the case if one plot the signal
    2. about half of the LAD records satisfy the "2-lead" criteria, but fail for the "3-lead" criteria, which means that their axis is (-30°, 0°) which is not truely LAD
    3. (Aug. 15, 2020; resolved, and changed to 1000) tranche F, the Georgia subset, has ADC gain 4880 which might be too high. Thus obtained voltages are too low. 1000 might be a suitable (correct) value of ADC gain for this tranche just as the other tranches.
    4. "E04603" (all leads), "E06072" (chest leads, epecially V1-V3), "E06909" (lead V2), "E07675" (lead V3), "E07941" (lead V6), "E08321" (lead V6) has exceptionally large values at rpeaks, reading (`load_data`) these two records using `wfdb` would bring in `nan` values. One can check using the following code
    >>> rec = "E04603"
    >>> dr.plot(rec, dr.load_data(rec, backend="scipy", units="uv"))  # currently raising error

    References:
    -----------
    [0] https://physionetchallenges.github.io/2021/
    [1] https://physionetchallenges.github.io/2020/
    [2] http://2018.icbeb.org/#
    [3] https://physionet.org/content/incartdb/1.0.0/
    [4] https://physionet.org/content/ptbdb/1.0.0/
    [5] https://physionet.org/content/ptb-xl/1.0.1/
    [6] (deprecated) https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/
    [7] (recommended) https://storage.cloud.google.com/physionetchallenge2021-public-datasets/
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
            print and log verbosity
        """
        super().__init__(db_name="CINC2021", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        
        self.rec_ext = "mat"
        self.ann_ext = "hea"

        self.db_tranches = list("ABCDEFG")
        self.tranche_names = ED({
            "A": "CPSC",
            "B": "CPSC-Extra",
            "C": "StPetersburg",
            "D": "PTB",
            "E": "PTB-XL",
            "F": "Georgia",
            "G": "CUSPHNFH",
        })
        self.rec_prefix = ED({
            "A": "A", "B": "Q", "C": "I", "D": "S", "E": "HR", "F": "E", "G": "JS",
        })

        self.db_dir_base = db_dir
        self.db_dirs = ED({tranche:"" for tranche in self.db_tranches})
        self._all_records = None
        self._stats = pd.DataFrame()
        self._stats_columns = {
            "record", "tranche", "tranche_name",
            "nb_leads", "fs", "nb_samples",
            "age", "sex",
            "medical_prescription", "history", "symptom_or_surgery",
            "diagnosis", "diagnosis_scored",  # in the form of abbreviations
        }
        self._ls_rec()  # loads file system structures into self.db_dirs and self._all_records
        self._aggregate_stats()

        self._diagnoses_records_list = None
        self._ls_diagnoses_records()

        self.fs = {
            "A": 500, "B": 500, "C": 257, "D": 1000, "E": 500, "F": 500,
        }
        self.spacing = {t: 1000 / f for t,f in self.fs.items()}

        self.all_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",]

        self.df_ecg_arrhythmia = dx_mapping_all[["Dx", "SNOMEDCTCode", "Abbreviation"]]
        self.ann_items = [
            "rec_name", "nb_leads", "fs", "nb_samples", "datetime", "age", "sex",
            "diagnosis", "df_leads",
            "medical_prescription", "history", "symptom_or_surgery",
        ]
        self.label_trans_dict = equiv_class_dict.copy()

        # self.value_correction_factor = ED({tranche:1 for tranche in self.db_tranches})
        # self.value_correction_factor.F = 4.88  # ref. ISSUES 3

        self.exceptional_records = ["E04603", "E06072", "E06909", "E07675", "E07941", "E08321"]  # ref. ISSUES 4


    def get_subject_id(self, rec:str) -> int:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        sid: int,
            the `subject_id` corr. to `rec`
        """
        s2d = {"A":"11", "B":"12", "C":"21", "D":"31", "E":"32", "F":"41", "G":"51",}
        s2d = {self.rec_prefix[k]:v for k,v in s2d.items()}
        prefix = "".join(re.findall(r"[A-Z]", rec))
        n = rec.replace(prefix,"")
        sid = int(f"{s2d[prefix]}{'0'*(8-len(n))}{n}")
        return sid

    
    def _ls_rec(self) -> NoReturn:
        """ finished, checked,

        list all the records and load into `self._all_records`,
        facilitating further uses
        """
        fn = "record_list.json"
        record_list_fp = os.path.join(self.db_dir_base, fn)
        if os.path.isfile(record_list_fp):
            with open(record_list_fp, "r") as f:
                self._all_records = json.load(f)
            for tranche in self.db_tranches:
                self.db_dirs[tranche] = os.path.join(self.db_dir_base, os.path.dirname(self._all_records[tranche][0]))
                self._all_records[tranche] = [os.path.basename(f) for f in self._all_records[tranche]]
        else:
            print("Please wait patiently to let the reader find all records of all the tranches...")
            start = time.time()
            rec_patterns_with_ext = {
                tranche: f"^{self.rec_prefix[tranche]}(?:\d+).{self.rec_ext}$" \
                    for tranche in self.db_tranches
            }
            self._all_records = \
                get_record_list_recursive3(self.db_dir_base, rec_patterns_with_ext)
            to_save = deepcopy(self._all_records)
            for tranche in self.db_tranches:
                tmp_dirname = [ os.path.dirname(f) for f in self._all_records[tranche] ]
                if len(set(tmp_dirname)) != 1:
                    if len(set(tmp_dirname)) > 1:
                        raise ValueError(f"records of tranche {tranche} are stored in several folders!")
                    else:
                        raise ValueError(f"no record found for tranche {tranche}!")
                self.db_dirs[tranche] = os.path.join(self.db_dir_base, tmp_dirname[0])
                self._all_records[tranche] = [os.path.basename(f) for f in self._all_records[tranche]]
            print(f"Done in {time.time() - start:.5f} seconds!")
            with open(os.path.join(self.db_dir_base, fn), "w") as f:
                json.dump(to_save, f)
        self._all_records = ED(self._all_records)


    def _aggregate_stats(self) -> NoReturn:
        """ finished, checked,

        aggregate stats on the whole dataset
        """
        stats_file = "stats.csv"
        list_sep = ";"
        stats_file_fp = os.path.join(self.db_dir_base, stats_file)
        if os.path.isfile(stats_file_fp):
            self._stats = pd.read_csv(stats_file_fp)
        if self._stats.empty or self._stats_columns != set(self._stats.columns):
            print("Please wait patiently to let the reader collect statistics on the whole dataset...")
            start = time.time()
            self._stats = pd.DataFrame(list_sum(self._all_records.values()), columns=["record"])
            self._stats["tranche"] = self._stats["record"].apply(lambda rec: self._get_tranche(rec))
            self._stats["tranche_name"] = self._stats["tranche"].apply(lambda t: self.tranche_names[t])
            for k in ["diagnosis", "diagnosis_scored",]:
                self._stats[k] = ""  # otherwise cells in the first row would be str instead of list
            for idx, row in self._stats.iterrows():
                ann_dict = self.load_ann(row["record"])
                for k in ["nb_leads", "fs", "nb_samples", "age", "sex", "medical_prescription", "history", "symptom_or_surgery",]:
                    self._stats.at[idx, k] = ann_dict[k]
                for k in ["diagnosis", "diagnosis_scored",]:
                    self._stats.at[idx, k] = ann_dict[k]["diagnosis_abbr"]
            _stats_to_save = self._stats.copy()
            for k in ["diagnosis", "diagnosis_scored",]:
                _stats_to_save[k] = _stats_to_save[k].apply(lambda l: list_sep.join(l))
            _stats_to_save.to_csv(stats_file_fp, index=False)
            print(f"Done in {time.time() - start:.5f} seconds!")
        else:
            for k in ["diagnosis", "diagnosis_scored",]:
                for idx, row in self._stats.iterrows():
                    self._stats.at[idx, k] = row[k].split(list_sep)


    @property
    def df_stats(self):
        """
        """
        return self._stats


    def _ls_diagnoses_records(self) -> NoReturn:
        """ finished, checked,

        list all the records for all diagnoses
        """
        fn = "diagnoses_records_list.json"
        dr_fp = os.path.join(self.db_dir_base, fn)
        if os.path.isfile(dr_fp):
            with open(dr_fp, "r") as f:
                self._diagnoses_records_list = json.load(f)
        else:
            print("Please wait several minutes patiently to let the reader list records for each diagnosis...")
            start = time.time()
            self._diagnoses_records_list = {d: [] for d in df_weights_abbr.columns.values.tolist()}
            if not self._stats.empty:
                for d in df_weights_abbr.columns.values.tolist():
                    self._diagnoses_records_list[d] = \
                        sorted(self._stats[self._stats["diagnosis_scored"].apply(lambda l: d in l)]["record"].tolist())
            else:
                for tranche, l_rec in self._all_records.items():
                    for rec in l_rec:
                        ann = self.load_ann(rec)
                        ld = ann["diagnosis_scored"]["diagnosis_abbr"]
                        for d in ld:
                            self._diagnoses_records_list[d].append(rec)
            print(f"Done in {time.time() - start:.5f} seconds!")
            with open(dr_fp, "w") as f:
                json.dump(self._diagnoses_records_list, f)
        self._diagnoses_records_list = ED(self._diagnoses_records_list)


    @property
    def diagnoses_records_list(self):
        """ finished, checked,
        """
        if self._diagnoses_records_list is None:
            self._ls_diagnoses_records()
        return self._diagnoses_records_list


    def _get_tranche(self, rec:str) -> str:
        """ finished, checked,

        get the tranche"s symbol (one of "A","B","C","D","E","F") of a record via its name

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        tranche, str,
            symbol of the tranche, ref. `self.rec_prefix`
        """
        prefix = "".join(re.findall(r"[A-Z]", rec))
        tranche = {v:k for k,v in self.rec_prefix.items()}[prefix]
        return tranche


    def get_data_filepath(self, rec:str, with_ext:bool=True) -> str:
        """ finished, checked,

        get the absolute file path of the data file of `rec`

        Parameters:
        -----------
        rec: str,
            name of the record
        with_ext: bool, default True,
            if True, the returned file path comes with file extension,
            otherwise without file extension,
            which is useful for `wfdb` functions

        Returns:
        --------
        fp: str,
            absolute file path of the data file of the record
        """
        tranche = self._get_tranche(rec)
        fp = os.path.join(self.db_dirs[tranche], f"{rec}.{self.rec_ext}")
        if not with_ext:
            fp = os.path.splitext(fp)[0]
        return fp

    
    def get_header_filepath(self, rec:str, with_ext:bool=True) -> str:
        """ finished, checked,

        get the absolute file path of the header file of `rec`

        Parameters:
        -----------
        rec: str,
            name of the record
        with_ext: bool, default True,
            if True, the returned file path comes with file extension,
            otherwise without file extension,
            which is useful for `wfdb` functions

        Returns:
        --------
        fp: str,
            absolute file path of the header file of the record
        """
        tranche = self._get_tranche(rec)
        fp = os.path.join(self.db_dirs[tranche], f"{rec}.{self.ann_ext}")
        if not with_ext:
            fp = os.path.splitext(fp)[0]
        return fp

    
    def get_ann_filepath(self, rec:str, with_ext:bool=True) -> str:
        """ finished, checked,
        alias for `get_header_filepath`
        """
        fp = self.get_header_filepath(rec, with_ext=with_ext)
        return fp


    def load_data(self, rec:str, leads:Optional[Union[str, List[str]]]=None, data_format:str="channel_first", backend:str="wfdb", units:str="mV", fs:Optional[Real]=None) -> np.ndarray:
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
        backend: str, default "wfdb",
            the backend data reader, can also be "scipy"
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        fs: real number, optional,
            if not None, the loaded data will be resampled to this frequency
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        assert data_format.lower() in ["channel_first", "lead_first", "channel_last", "lead_last"]
        tranche = self._get_tranche(rec)
        if not leads:
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads
        if backend.lower() == "wfdb":
            rec_fp = self.get_data_filepath(rec, with_ext=False)
            # p_signal of "lead_last" format
            wfdb_rec = wfdb.rdrecord(rec_fp, physical=True, channel_names=_leads)
            data = np.asarray(wfdb_rec.p_signal.T)
            # lead_units = np.vectorize(lambda s: s.lower())(wfdb_rec.units)
        elif backend.lower() == "scipy":
            # loadmat of "lead_first" format
            rec_fp = self.get_data_filepath(rec, with_ext=True)
            data = loadmat(rec_fp)["val"]
            header_info = self.load_ann(rec, raw=False)["df_leads"]
            baselines = header_info["baseline"].values.reshape(data.shape[0], -1)
            adc_gain = header_info["adc_gain"].values.reshape(data.shape[0], -1)
            data = np.asarray(data-baselines) / adc_gain
            leads_ind = [self.all_leads.index(item) for item in _leads]
            data = data[leads_ind,:]
            # lead_units = np.vectorize(lambda s: s.lower())(header_info["df_leads"]["adc_units"].values)
        else:
            raise ValueError(f"backend `{backend.lower()}` not supported for loading data")
        
        # ref. ISSUES 3, for multiplying `value_correction_factor`
        # data = data * self.value_correction_factor[tranche]

        if units.lower() in ["uv", "μv"]:
            data = data * 1000

        rec_fs = self.get_fs(rec, from_hea=True)
        if fs is not None and fs != rec_fs:
            data = resample_poly(data, fs, rec_fs, axis=1)
        # if fs is not None and fs != self.fs[tranche]:
        #     data = resample_poly(data, fs, self.fs[tranche], axis=1)

        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T

        return data

    
    def load_ann(self, rec:str, raw:bool=False, backend:str="wfdb") -> Union[dict,str]:
        """ finished, checked,

        load annotations (header) stored in the .hea files
        
        Parameters:
        -----------
        rec: str,
            name of the record
        raw: bool, default False,
            if True, the raw annotations without parsing will be returned
        backend: str, default "wfdb", case insensitive,
            if is "wfdb", `wfdb.rdheader` will be used to load the annotations;
            if is "naive", annotations will be parsed from the lines read from the header files
        
        Returns:
        --------
        ann_dict, dict or str,
            the annotations with items: ref. `self.ann_items`
        """
        tranche = self._get_tranche(rec)
        ann_fp = self.get_ann_filepath(rec, with_ext=True)
        with open(ann_fp, "r") as f:
            header_data = f.read().splitlines()
        
        if raw:
            ann_dict = "\n".join(header_data)
            return ann_dict

        if backend.lower() == "wfdb":
            ann_dict = self._load_ann_wfdb(rec, header_data)
        elif backend.lower() == "naive":
            ann_dict = self._load_ann_naive(header_data)
        else:
            raise ValueError(f"backend `{backend.lower()}` not supported for loading annotations")
        return ann_dict


    def _load_ann_wfdb(self, rec:str, header_data:List[str]) -> dict:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record
        header_data: list of str,
            list of lines read directly from a header file,
            complementary to data read using `wfdb.rdheader` if applicable,
            this data will be used, since `datetime` is not well parsed by `wfdb.rdheader`

        Returns:
        --------
        ann_dict, dict,
            the annotations with items: ref. `self.ann_items`
        """
        header_fp = self.get_header_filepath(rec, with_ext=False)
        header_reader = wfdb.rdheader(header_fp)
        ann_dict = {}
        ann_dict["rec_name"], ann_dict["nb_leads"], ann_dict["fs"], ann_dict["nb_samples"], ann_dict["datetime"], daytime = header_data[0].split(" ")

        ann_dict["nb_leads"] = int(ann_dict["nb_leads"])
        ann_dict["fs"] = int(ann_dict["fs"])
        ann_dict["nb_samples"] = int(ann_dict["nb_samples"])
        ann_dict["datetime"] = datetime.strptime(" ".join([ann_dict["datetime"], daytime]), "%d-%b-%Y %H:%M:%S")
        try: # see NOTE. 1.
            ann_dict["age"] = int([l for l in header_reader.comments if "Age" in l][0].split(": ")[-1])
        except:
            ann_dict["age"] = np.nan
        try:
            ann_dict["sex"] = [l for l in header_reader.comments if "Sex" in l][0].split(": ")[-1]
        except:
            ann_dict["sex"] = "Unknown"
        try:
            ann_dict["medical_prescription"] = [l for l in header_reader.comments if "Rx" in l][0].split(": ")[-1]
        except:
            ann_dict["medical_prescription"] = "Unknown"
        try:
            ann_dict["history"] = [l for l in header_reader.comments if "Hx" in l][0].split(": ")[-1]
        except:
            ann_dict["history"] = "Unknown"
        try:
            ann_dict["symptom_or_surgery"] = [l for l in header_reader.comments if "Sx" in l][0].split(": ")[-1]
        except:
            ann_dict["symptom_or_surgery"] = "Unknown"

        l_Dx = [l for l in header_reader.comments if "Dx" in l][0].split(": ")[-1].split(",")
        ann_dict["diagnosis"], ann_dict["diagnosis_scored"] = self._parse_diagnosis(l_Dx)

        df_leads = pd.DataFrame()
        for k in ["file_name", "fmt", "byte_offset", "adc_gain", "units", "adc_res", "adc_zero", "baseline", "init_value", "checksum", "block_size", "sig_name"]:
            df_leads[k] = header_reader.__dict__[k]
        df_leads = df_leads.rename(
            columns={
                "sig_name": "lead_name",
                "units":"adc_units",
                "file_name":"filename",
            }
        )
        df_leads.index = df_leads["lead_name"]
        df_leads.index.name = None
        ann_dict["df_leads"] = df_leads

        return ann_dict


    def _load_ann_naive(self, header_data:List[str]) -> dict:
        """ finished, checked,

        load annotations (header) using raw data read directly from a header file
        
        Parameters:
        -----------
        header_data: list of str,
            list of lines read directly from a header file
        
        Returns:
        --------
        ann_dict, dict,
            the annotations with items: ref. `self.ann_items`
        """
        ann_dict = {}
        ann_dict["rec_name"], ann_dict["nb_leads"], ann_dict["fs"], ann_dict["nb_samples"], ann_dict["datetime"], daytime = header_data[0].split(" ")

        ann_dict["nb_leads"] = int(ann_dict["nb_leads"])
        ann_dict["fs"] = int(ann_dict["fs"])
        ann_dict["nb_samples"] = int(ann_dict["nb_samples"])
        ann_dict["datetime"] = datetime.strptime(" ".join([ann_dict["datetime"], daytime]), "%d-%b-%Y %H:%M:%S")
        try: # see NOTE. 1.
            ann_dict["age"] = int([l for l in header_data if l.startswith("#Age")][0].split(": ")[-1])
        except:
            ann_dict["age"] = np.nan
        try:
            ann_dict["sex"] = [l for l in header_data if l.startswith("#Sex")][0].split(": ")[-1]
        except:
            ann_dict["sex"] = "Unknown"
        try:
            ann_dict["medical_prescription"] = [l for l in header_data if l.startswith("#Rx")][0].split(": ")[-1]
        except:
            ann_dict["medical_prescription"] = "Unknown"
        try:
            ann_dict["history"] = [l for l in header_data if l.startswith("#Hx")][0].split(": ")[-1]
        except:
            ann_dict["history"] = "Unknown"
        try:
            ann_dict["symptom_or_surgery"] = [l for l in header_data if l.startswith("#Sx")][0].split(": ")[-1]
        except:
            ann_dict["symptom_or_surgery"] = "Unknown"

        l_Dx = [l for l in header_data if l.startswith("#Dx")][0].split(": ")[-1].split(",")
        ann_dict["diagnosis"], ann_dict["diagnosis_scored"] = self._parse_diagnosis(l_Dx)

        ann_dict["df_leads"] = self._parse_leads(header_data[1:13])

        return ann_dict


    def _parse_diagnosis(self, l_Dx:List[str]) -> Tuple[dict, dict]:
        """ finished, checked,

        Parameters:
        -----------
        l_Dx: list of str,
            raw information of diagnosis, read from a header file

        Returns:
        --------
        diag_dict:, dict,
            diagnosis, including SNOMED CT Codes, fullnames and abbreviations of each diagnosis
        diag_scored_dict: dict,
            the scored items in `diag_dict`
        """
        diag_dict, diag_scored_dict = {}, {}
        # try:
        diag_dict["diagnosis_code"] = [item for item in l_Dx if item in dx_mapping_all["SNOMEDCTCode"].tolist()]
        # in case not listed in dx_mapping_all
        left = [item for item in l_Dx if item not in dx_mapping_all["SNOMEDCTCode"].tolist()]
        # selection = dx_mapping_all["SNOMEDCTCode"].isin(diag_dict["diagnosis_code"])
        # diag_dict["diagnosis_abbr"] = dx_mapping_all[selection]["Abbreviation"].tolist()
        # diag_dict["diagnosis_fullname"] = dx_mapping_all[selection]["Dx"].tolist()
        diag_dict["diagnosis_abbr"] = \
            [ dx_mapping_all[dx_mapping_all["SNOMEDCTCode"]==dc]["Abbreviation"].values[0] \
                for dc in diag_dict["diagnosis_code"] ] + left
        diag_dict["diagnosis_fullname"] = \
            [ dx_mapping_all[dx_mapping_all["SNOMEDCTCode"]==dc]["Dx"].values[0] \
                for dc in diag_dict["diagnosis_code"] ] + left
        diag_dict["diagnosis_code"] = diag_dict["diagnosis_code"] + left
        scored_indices = np.isin(
            diag_dict["diagnosis_code"],
            dx_mapping_scored["SNOMEDCTCode"].values
        )
        diag_scored_dict["diagnosis_code"] = \
            [ item for idx, item in enumerate(diag_dict["diagnosis_code"]) \
                if scored_indices[idx] ]
        diag_scored_dict["diagnosis_abbr"] = \
            [ item for idx, item in enumerate(diag_dict["diagnosis_abbr"]) \
                if scored_indices[idx] ]
        diag_scored_dict["diagnosis_fullname"] = \
            [ item for idx, item in enumerate(diag_dict["diagnosis_fullname"]) \
                if scored_indices[idx] ]
        # except:  # the old version, the Dx's are abbreviations, deprecated
            # diag_dict["diagnosis_abbr"] = diag_dict["diagnosis_code"]
            # selection = dx_mapping_all["Abbreviation"].isin(diag_dict["diagnosis_abbr"])
            # diag_dict["diagnosis_fullname"] = dx_mapping_all[selection]["Dx"].tolist()
        # if not keep_original:
        #     for idx, d in enumerate(ann_dict["diagnosis_abbr"]):
        #         if d in ["Normal", "NSR"]:
        #             ann_dict["diagnosis_abbr"] = ["N"]
        return diag_dict, diag_scored_dict


    def _parse_leads(self, l_leads_data:List[str]) -> pd.DataFrame:
        """ finished, checked,

        Parameters:
        -----------
        l_leads_data: list of str,
            raw information of each lead, read from a header file

        Returns:
        --------
        df_leads: DataFrame,
            infomation of each leads in the format of DataFrame
        """
        df_leads = pd.read_csv(io.StringIO("\n".join(l_leads_data)), delim_whitespace=True, header=None)
        df_leads.columns = ["filename", "fmt+byte_offset", "adc_gain+units", "adc_res", "adc_zero", "init_value", "checksum", "block_size", "lead_name",]
        df_leads["fmt"] = df_leads["fmt+byte_offset"].apply(lambda s: s.split("+")[0])
        df_leads["byte_offset"] = df_leads["fmt+byte_offset"].apply(lambda s: s.split("+")[1])
        df_leads["adc_gain"] = df_leads["adc_gain+units"].apply(lambda s: s.split("/")[0])
        df_leads["adc_units"] = df_leads["adc_gain+units"].apply(lambda s: s.split("/")[1])
        for k in ["byte_offset", "adc_gain", "adc_res", "adc_zero", "init_value", "checksum",]:
            df_leads[k] = df_leads[k].apply(lambda s: int(s))
        df_leads["baseline"] = df_leads["adc_zero"]
        df_leads = df_leads[["filename", "fmt", "byte_offset", "adc_gain", "adc_units", "adc_res", "adc_zero", "baseline", "init_value", "checksum", "block_size", "lead_name"]]
        df_leads.index = df_leads["lead_name"]
        df_leads.index.name = None
        return df_leads


    def load_header(self, rec:str, raw:bool=False) -> Union[dict,str]:
        """
        alias for `load_ann`, as annotations are also stored in header files
        """
        return self.load_ann(rec, raw)

    
    def get_labels(self, rec:str, scored_only:bool=True, fmt:str="s", normalize:bool=True) -> List[str]:
        """ finished, checked,

        read labels (diagnoses or arrhythmias) of a record
        
        Parameters:
        -----------
        rec: str,
            name of the record
        scored_only: bool, default True,
            only get the labels that are scored in the CINC2020 official phase
        fmt: str, default "a",
            the format of labels, one of the following (case insensitive):
            - "a", abbreviations
            - "f", full names
            - "s", SNOMEDCTCode
        normalize: bool, default True,
            if True, the labels will be transformed into their equavalents,
            which are defined in `utils.utils_misc.cinc2020_aux_data.py`,
            and duplicates would be removed if exist after normalization
        
        Returns:
        --------
        labels, list,
            the list of labels
        """
        ann_dict = self.load_ann(rec)
        if scored_only:
            _labels = ann_dict["diagnosis_scored"]
        else:
            _labels = ann_dict["diagnosis"]
        if fmt.lower() == "a":
            _labels = _labels["diagnosis_abbr"]
        elif fmt.lower() == "f":
            _labels = _labels["diagnosis_fullname"]
        elif fmt.lower() == "s":
            _labels = _labels["diagnosis_code"]
        else:
            raise ValueError(f"`fmt` should be one of `a`, `f`, `s`, but got `{fmt}`")
        if normalize:
            # labels = [self.label_trans_dict.get(item, item) for item in labels]
            # remove possible duplicates after normalization
            labels = []
            for item in _labels:
                new_item = self.label_trans_dict.get(item, item)
                if new_item not in labels:
                    labels.append(new_item)
        else:
            labels = _labels
        return labels


    def get_fs(self, rec:str, from_hea:bool=True) -> Real:
        """ finished, checked,

        get the sampling frequency of a record

        Parameters:
        -----------
        rec: str,
            name of the record
        from_hea: bool, default True,
            if True, get sampling frequency from corresponding header file of the record;
            otherwise from `self.fs`

        Returns:
        --------
        fs: real number,
            sampling frequency of the record `rec`
        """
        if from_hea:
            fs = self.load_ann(rec)["fs"]
        else:
            tranche = self._get_tranche(rec)
            fs = self.fs[tranche]
        return fs

    
    def get_subject_info(self, rec:str, items:Optional[List[str]]=None) -> dict:
        """ finished, checked,

        read auxiliary information of a subject (a record) stored in the header files

        Parameters:
        -----------
        rec: str,
            name of the record
        items: list of str, optional,
            items of the subject's information (e.g. sex, age, etc.)
        
        Returns:
        --------
        subject_info: dict,
            information about the subject, including
            "age", "sex", "medical_prescription", "history", "symptom_or_surgery",
        """
        if items is None or len(items) == 0:
            info_items = [
                "age", "sex", "medical_prescription", "history", "symptom_or_surgery",
            ]
        else:
            info_items = items
        ann_dict = self.load_ann(rec)
        subject_info = [ann_dict[item] for item in info_items]

        return subject_info


    def save_challenge_predictions(self, rec:str, output_dir:str, scores:List[Real], labels:List[int], classes:List[str]) -> NoReturn:
        """ NOT finished, NOT checked, need updating, 
        
        TODO: update for the official phase

        Parameters:
        -----------
        rec: str,
            name of the record
        output_dir: str,
            directory to save the predictions
        scores: list of real,
            raw predictions
        labels: list of int,
            0 or 1, binary predictions
        classes: list of str,
            SNOMEDCTCode of binary predictions
        """
        new_file = f"{rec}.csv"
        output_file = os.path.join(output_dir, new_file)

        # Include the filename as the recording number
        recording_string = f"#{rec}"
        class_string = ",".join(classes)
        label_string = ",".join(str(i) for i in labels)
        score_string = ",".join(str(i) for i in scores)

        with open(output_file, "w") as f:
            # f.write(recording_string + "\n" + class_string + "\n" + label_string + "\n" + score_string + "\n")
            f.write("\n".join([recording_string, class_string, label_string, score_string, ""]))


    def plot(self, rec:str, data:Optional[np.ndarray]=None, ann:Optional[Dict[str, np.ndarray]]=None, ticks_granularity:int=0, leads:Optional[Union[str, List[str]]]=None, same_range:bool=False, waves:Optional[Dict[str, Sequence[int]]]=None, **kwargs) -> NoReturn:
        """ finished, checked, to improve,

        plot the signals of a record or external signals (units in μV),
        with metadata (fs, labels, tranche, etc.),
        possibly also along with wave delineations

        Parameters:
        -----------
        rec: str,
            name of the record
        data: ndarray, optional,
            (12-lead) ecg signal to plot,
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
        same_range: bool, default False,
            if True, forces all leads to have the same y range
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
        `Locator` of `plt` has default `MAXTICKS` equal to 1000,
        if not modifying this number, at most 40 seconds of signal could be plotted once

        Contributors: Jeethan, and WEN Hao
        """
        tranche = self._get_tranche(rec)
        if tranche in "CDE":
            physionet_lightwave_suffix = ED({
                "C": "incartdb/1.0.0",
                "D": "ptbdb/1.0.0",
                "E": "ptb-xl/1.0.1",
            })
            url = f"https://physionet.org/lightwave/?db={physionet_lightwave_suffix[tranche]}"
            print(f"better view: {url}")
            
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

        # lead_list = self.load_ann(rec)["df_leads"]["lead_name"].tolist()
        # lead_indices = [lead_list.index(l) for l in _leads]
        lead_indices = [self.all_leads.index(l) for l in _leads]
        if data is None:
            _data = self.load_data(rec, data_format="channel_first", units="μV")[lead_indices]
        else:
            units = self._auto_infer_units(data)
            print(f"input data is auto detected to have units in {units}")
            if units.lower() == "mv":
                _data = 1000 * data
            else:
                _data = data
            assert _data.shape[0] == len(_leads), \
                f"number of leads from data of shape ({_data.shape[0]}) does not match the length ({len(_leads)}) of `leads`"
        
        if same_range:
            y_ranges = np.ones((_data.shape[0],)) * np.max(np.abs(_data)) + 100
        else:
            y_ranges = np.max(np.abs(_data), axis=1) + 100

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
        palette = {"p_waves": "green", "qrs": "red", "t_waves": "pink",}
        plot_alpha = 0.4

        if ann is None or data is None:
            diag_scored = self.get_labels(rec, scored_only=True, fmt="a")
            diag_all = self.get_labels(rec, scored_only=False, fmt="a")
        else:
            diag_scored = ann["scored"]
            diag_all = ann["all"]

        nb_leads = len(_leads)

        seg_len = self.fs[tranche] * 25  # 25 seconds
        nb_segs = _data.shape[1] // seg_len

        t = np.arange(_data.shape[1]) / self.fs[tranche]
        duration = len(t) / self.fs[tranche]
        fig_sz_w = int(round(4.8 * duration))
        fig_sz_h = 6 * y_ranges / 1500
        fig, axes = plt.subplots(nb_leads, 1, sharex=True, figsize=(fig_sz_w, np.sum(fig_sz_h)))
        if nb_leads == 1:
            axes = [axes]
        for idx in range(nb_leads):
            axes[idx].plot(t, _data[idx], color="black", label=f"lead - {_leads[idx]}")
            axes[idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
            # NOTE that `Locator` has default `MAXTICKS` equal to 1000
            if ticks_granularity >= 1:
                axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                axes[idx].grid(which="major", linestyle="-", linewidth="0.5", color="red")
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(which="minor", linestyle=":", linewidth="0.5", color="black")
            # add extra info. to legend
            # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
            axes[idx].plot([], [], " ", label=f"labels_s - {','.join(diag_scored)}")
            axes[idx].plot([], [], " ", label=f"labels_a - {','.join(diag_all)}")
            axes[idx].plot([], [], " ", label=f"tranche - {self.tranche_names[tranche]}")
            axes[idx].plot([], [], " ", label=f"fs - {self.fs[tranche]}")
            for w in ["p_waves", "qrs", "t_waves"]:
                for itv in eval(w):
                    axes[idx].axvspan(itv[0], itv[1], color=palette[w], alpha=plot_alpha)
            axes[idx].legend(loc="upper left")
            axes[idx].set_xlim(t[0], t[-1])
            axes[idx].set_ylim(-y_ranges[idx], y_ranges[idx])
            axes[idx].set_xlabel("Time [s]")
            axes[idx].set_ylabel("Voltage [μV]")
        plt.subplots_adjust(hspace=0.2)
        plt.show()


    def get_tranche_class_distribution(self, tranches:Sequence[str], scored_only:bool=True) -> Dict[str, int]:
        """ finished, checked,

        Parameters:
        -----------
        tranches: sequence of str,
            tranche symbols (A-F)
        scored_only: bool, default True,
            only get class distributions that are scored in the CINC2020 official phase
        
        Returns:
        --------
        distribution: dict,
            keys are abbrevations of the classes, values are appearance of corr. classes in the tranche.
        """
        tranche_names = [self.tranche_names[t] for t in tranches]
        df = dx_mapping_scored if scored_only else dx_mapping_all
        distribution = ED()
        for _, row in df.iterrows():
            num = (row[[tranche_names]].values).sum()
            if num > 0:
                distribution[row["Abbreviation"]] = num
        return distribution


    @staticmethod
    def get_arrhythmia_knowledge(arrhythmias:Union[str,List[str]], **kwargs) -> NoReturn:
        """ finished, checked,

        knowledge about ECG features of specific arrhythmias,

        Parameters:
        -----------
        arrhythmias: str, or list of str,
            the arrhythmia(s) to check, in abbreviations or in SNOMEDCTCode
        """
        if isinstance(arrhythmias, str):
            d = [normalize_class(arrhythmias)]
        else:
            d = [normalize_class(c) for c in arrhythmias]
        # pp = pprint.PrettyPrinter(indent=4)
        # unsupported = [item for item in d if item not in dx_mapping_all["Abbreviation"]]
        unsupported = [item for item in d if item not in dx_mapping_scored["Abbreviation"].values]
        assert len(unsupported) == 0, \
            f"`{unsupported}` {'is' if len(unsupported)==1 else 'are'} not supported!"
        for idx, item in enumerate(d):
            # pp.pprint(eval(f"EAK.{item}"))
            print(dict_to_str(eval(f"EAK.{item}")))
            if idx < len(d)-1:
                print("*"*110)


    def load_resampled_data(self, rec:str, data_format:str="channel_first", siglen:Optional[int]=None) -> np.ndarray:
        """ finished, checked,

        resample the data of `rec` to 500Hz,
        or load the resampled data in 500Hz, if the corr. data file already exists

        Parameters:
        -----------
        rec: str,
            name of the record
        data_format: str, default "channel_first",
            format of the ecg data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        siglen: int, optional,
            signal length, units in number of samples,
            if set, signal with length longer will be sliced to the length of `siglen`
            used for example when preparing/doing model training

        Returns:
        --------
        data: ndarray,
            the resampled (and perhaps sliced) signal data
        """
        tranche = self._get_tranche(rec)
        if siglen is None:
            rec_fp = os.path.join(self.db_dirs[tranche], f"{rec}_500Hz.npy")
        else:
            rec_fp = os.path.join(self.db_dirs[tranche], f"{rec}_500Hz_siglen_{siglen}.npy")
        if not os.path.isfile(rec_fp):
            # print(f"corresponding file {os.basename(rec_fp)} does not exist")
            data = self.load_data(rec, data_format="channel_first", units="mV", fs=None)
            rec_fs = self.get_fs(rec, from_hea=True)
            if rec_fs != 500:
                data = resample_poly(data, 500, rec_fs, axis=1)
            # if self.fs[tranche] != 500:
            #     data = resample_poly(data, 500, self.fs[tranche], axis=1)
            if siglen is not None and data.shape[1] >= siglen:
                # slice_start = (data.shape[1] - siglen)//2
                # slice_end = slice_start + siglen
                # data = data[..., slice_start:slice_end]
                data = ensure_siglen(data, siglen=siglen, fmt="channel_first")
                np.save(rec_fp, data)
            elif siglen is None:
                np.save(rec_fp, data)
        else:
            # print(f"loading from local file...")
            data = np.load(rec_fp)
        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T
        return data


    def load_raw_data(self, rec:str, backend:str="scipy") -> np.ndarray:
        """ finished, checked,

        load raw data from corresponding files with no further processing,
        in order to facilitate feeding data into the `run_12ECG_classifier` function

        Parameters:
        -----------
        rec: str,
            name of the record
        backend: str, default "scipy",
            the backend data reader, can also be "wfdb",
            note that "scipy" provides data in the format of "lead_first",
            while "wfdb" provides data in the format of "lead_last",

        Returns:
        --------
        raw_data: ndarray,
            raw data (d_signal) loaded from corresponding data file,
            without subtracting baseline nor dividing adc gain
        """
        tranche = self._get_tranche(rec)
        if backend.lower() == "wfdb":
            rec_fp = self.get_data_filepath(rec, with_ext=False)
            wfdb_rec = wfdb.rdrecord(rec_fp, physical=False)
            raw_data = np.asarray(wfdb_rec.d_signal)
        elif backend.lower() == "scipy":
            rec_fp = self.get_data_filepath(rec, with_ext=True)
            raw_data = loadmat(rec_fp)["val"]
        return raw_data


    def _check_nan(self, tranches:Union[str, Sequence[str]]) -> NoReturn:
        """ finished, checked,

        check if records from `tranches` has nan values

        accessing data using `p_signal` of `wfdb` would produce nan values,
        if exceptionally large values are encountered,
        this could help detect abnormal records as well

        Parameters:
        -----------
        tranches: str or sequence of str,
            tranches to check
        """
        for t in tranches:
            for rec in self.all_records[t]:
                data = self.load_data(rec)
                if np.isnan(data).any():
                    print(f"record {rec} from tranche {t} has nan values")



def prepare_dataset(input_directory:str,
                    output_directory:Optional[str]=None,
                    tranches:Optional[Sequence[str]]=None,
                    verbose:bool=False) -> NoReturn:
    """ finished, checked,

    Parameters:
    -----------
    input_directory: str,
        directory containing the .tar.gz files of the records and headers
    output_directory: str, optional,
        directory to store the extracted records and headers, under specific organization,
        if not specified, defaults to `input_directory`
    tranches: sequence of str, optional,
        the tranches to extract
    verbose: bool, default False,
        printint verbosity

    NOTE: currently, for updating headers only, corresponding .tar.gz file of records should be presented
    """
    import shutil, tarfile
    from glob import glob

    data_files = [
        "WFDB_CPSC2018.tar.gz",
        "WFDB_CPSC2018_2.tar.gz",
        "WFDB_StPetersburg.tar.gz",
        "WFDB_PTB.tar.gz",
        "WFDB_PTBXL.tar.gz",
        "WFDB_Ga.tar.gz",
        "WFDB_ShaoxingUniv.tar.gz",
        "WFDB_ChapmanShaoxing.tar.gz",
        "WFDB_Ningbo.tar.gz",
    ]
    header_files = [
        "CPSC2018-Headers.tar.gz",
        "CPSC2018-2-Headers.tar.gz",
        "StPetersburg-Headers.tar.gz",
        "PTB-Headers.tar.gz",
        "PTB-XL-Headers.tar.gz",
        "Ga-Headers.tar.gz",
        "ShaoxingUniv_Headers.tar.gz",
        "ChapmanShaoxing-Headers.tar.gz",
        "Ningbo-Headers.tar.gz",
    ]
    _tranches = "CPSC,CPSC_Extra,StPetersburg,PTB,PTB_XL,Georgia,CUSPHNFH".split(",")

    _dir = os.path.abspath(input_directory)
    # ShaoxingUniv (CUSPHNFH) is the union of ChapmanShaoxing and Ningbo
    if data_files[-3] in os.listdir(input_directory):
        flag_CUSPHNFH = False
        _data_files =  data_files[:-2]
        _header_files = header_files[:-2]
    else:
        flag_CUSPHNFH = True
        _data_files = deepcopy(data_files)
        _header_files = deepcopy(header_files)
    _data_files = \
        [os.path.basename(item) for item in glob(os.path.join(_dir, "WFDB_*.tar.gz")) if os.path.basename(item) in _data_files]
    _header_files = \
        [os.path.basename(item) for item in glob(os.path.join(_dir, "*Headers.tar.gz")) if os.path.basename(item) in _header_files]
    _output_directory = output_directory or input_directory
    assert all([header_files[data_files.index(item)] in _header_files for item in _data_files]), \
        "header files corresponding to some data files not found"

    if flag_CUSPHNFH:
        os.makedirs(os.path.join(_output_directory, "WFDB_CUSPHNFH"), exist_ok=True)

    acc = 0
    for i, df in enumerate(_data_files):
        if tranches and _tranches[data_files.index(df)] not in tranches:
            continue
        acc += 1
        if df in ["WFDB_ChapmanShaoxing.tar.gz", "WFDB_Ningbo.tar.gz",]:
            df_name = "WFDB_CUSPHNFH"
        else:
            df_name = df.replace(".tar.gz", "")
        if len(glob(os.path.join(_output_directory, df_name, "*.mat"))) > 0:
            pass
        else:
            with tarfile.open(os.path.join(_dir, df), "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        member.name = os.path.basename(member.name)
                        # header files will not be extracted,
                        # instead, they will be extracted from corresponding headers-only .tar.gz file
                        if os.path.splitext(member.name)[1] == ".hea":
                            continue
                        tar.extract(member, os.path.join(_output_directory, df_name))
                        if verbose:
                            print(f"extracted '{os.path.join(_output_directory, df_name, member.name)}'")
        print(f"finish extracting {df}")
        time.sleep(3)
        # corresponding header files
        hf = header_files[data_files.index(df)]
        with tarfile.open(os.path.join(_dir, hf), "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    member.name = os.path.basename(member.name)
                    tar.extract(member, os.path.join(_output_directory, df_name))
                    if verbose:
                        print(f"extracted '{os.path.join(_output_directory, df_name, member.name)}'")
        print(f"finish extracting {hf}")
        print(f"{df_name} done! --- {acc}/{len(tranches) if tranches else len(_data_files)}")
        if i < len(_data_files) - 1:
            time.sleep(3)


def get_parser() -> dict:
    """
    """
    import argparse
    description = "Prepare the dataset, uncompressing the .tar.gz files, and replacing the header files."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input_directory", type=str, required=True,
        help="input directory, containing .tar.gz files of records and headers",
        dest="input_directory",
    )
    parser.add_argument(
        "-o", "--output_directory", type=str,
        help="output directory",
        dest="output_directory",
    )
    parser.add_argument(
        "-t", "--tranches", type=str,
        help=f"""list of tranches, a subset of {",".join(_tranches)}, separated by comma""",
        dest="tranches",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help=f"verbosity",
        dest="verbose",
    )

    args = vars(parser.parse_args())

    return args



if __name__ == "__main__":
    # usage example:
    # python prepare_dataset.py --help (check for details of arguments)
    # python prepare_dataset.py -i "E:/Data/CinC2021/" -t "PTB,Georgia" -v
    args = get_parser()
    input_directory = args.get("input_directory")
    output_directory = args.get("output_directory", None)
    tranches = args.get("tranches", None)
    verbose = args.get("verbose", False)
    if tranches:
        tranches = tranches.split(",")
    run(input_directory, output_directory, tranches, verbose)
