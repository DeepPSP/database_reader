# -*- coding: utf-8 -*-
"""
"""
import os
from datetime import datetime
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

import wfdb
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from ..base import PhysioNetDataBase


__all__ = [
    "LTSTDB",
]


class LTSTDB(PhysioNetDataBase):
    """ Finished, to be improved,

    Long Term ST Database

    ABOUT ltstdb:
    -------------
    1. contains 86 lengthy ECG recordings of 80 human subjects
    2. all records are between 21 and 24 hours in duration, and contain two or three ECG signals
    3. digitized at 250 samples per second with 12-bit resolution over a range of ±10 millivolts
    4. exhibits a variety of events of ST segment changes, including ischemic ST episodes, axis-related non-ischemic ST episodes, episodes of slow ST level drift, and episodes containing mixtures of these phenomena
    5. each record includes a set of meticulously verified ST episode and signal quality annotations, together with additional beat-by-beat QRS annotations and ST level measurements
    6. for annotations: experts examine the time series of ST level measurements in order to locate and to mark a set of local reference points, which are used to construct a piecewise linear baseline ST level function
    7. measurements in .16a files were used to construct ST level and deviation functions for each signal

    NOTE:
    -----

    ISSUES:
    -------

    Usage:
    ------
    1. ST segment

    References:
    -----------
    [1] https://physionet.org/content/ltstdb/1.0.0/
    """
    def __init__(self, db_dir:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="ltstdb", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.fs = 250
        
        self.data_ext = "dat"
        self.ann_ext = "atr"
        self.all_anno_extensions = ["ari", "atr", "16a", "sta", "stb", "stc"]
        """
        1. ari (automatically-generated beat annotations)
        2. atr (manually corrected beat annotations)
        3. 16a (automatically-generated, manually-corrected ST-segment measurements, based on 16-second moving averages, updated for each beat)
        4. sta (ST-segment episode annotations, Vmin = 75 µV, Tmin = 30 s)
        5. stb (ST-segment episode annotations, Vmin = 100 µV, Tmin = 30 s)
        6. stc (ST-segment episode annotations, Vmin = 100 µV, Tmin = 60 s)
        """

        self._ls_rec()
        
        self.all_leads = ["ML2", "A-I", "ECG", "II", "MV2", "V3", "V6", "aVF", "E-S", "A-S", "V4", "MLIII", "V2", "V5"]
        self.episode_protocols = ["a", "b", "c"]
        self.all_urd_intervals = [
            {
                "record": "s20291",
                "lead_number": 0,
                "urd_intervals": [{"start_index": 4144500, "end_index": 4174000}]
            },
            {
                "record": "s20291",
                "lead_number": 1,
                "urd_intervals": [{"start_index": 4144500, "end_index": 4174000}]
            },
            {
                "record": "s20561",
                "lead_number": 0,
                "urd_intervals":[
                    {"start_index": 5869000, "end_index": 8785500},
                    {"start_index": 20058500, "end_index": 20582000}
                ]
            },
            {
                "record": "s20561",
                "lead_number": 1,
                "urd_intervals": [
                    {"start_index": 5869000, "end_index": 8785500},
                    {"start_index": 20058500, "end_index": 20582000}
                ]
            },
            {
                "record": "s20571",
                "lead_number": 0,
                "urd_intervals": [
                    {"start_index": 8909000, "end_index": 9125500},
                    {"start_index": 10778500, "end_index": 16876500}
                ]
            },
            {
                "record": "s20571",
                "lead_number": 1,
                "urd_intervals": [
                    {"start_index": 8909000, "end_index": 9125500},
                    {"start_index": 10778500, "end_index": 16876500}
                ]
            },
            {
                "record": "s20601",
                "lead_number": 0,
                "urd_intervals": [{"start_index": 21133000, "end_index": 21283000}]
            },
            {
                "record": "s20601",
                "lead_number": 1,
                "urd_intervals": [
                    {"start_index": 20139500, "end_index": 20205000},
                    {"start_index": 21133000, "end_index": 21283000}
                ]
            },
            {
                "record": "s20621",
                "lead_number": 0,
                "urd_intervals": [
                    {"start_index": 2613000, "end_index": 2881000},
                    {"start_index": 8556500, "end_index": 8569500},
                    {"start_index": 9182500, "end_index": 9245500},
                    {"start_index": 12287500, "end_index": 12433500},
                    {"start_index": 18417000, "end_index": 18456500},
                    {"start_index": 20386500, "end_index": 20439000},
                    {"start_index": 20507500, "end_index": 20527500}
                ]
            },
            {
                "record": "s20621",
                "lead_number": 1,
                "urd_intervals": [
                    {"start_index": 2613000, "end_index": 2881000},
                    {"start_index": 8556500, "end_index": 8569500},
                    {"start_index": 9182500, "end_index": 9245500},
                    {"start_index": 12225500, "end_index": 12433500},
                    {"start_index": 18417000, "end_index": 18456500},
                    {"start_index": 20386500, "end_index": 20439000},
                    {"start_index": 20507500, "end_index": 20527500}
                ]
            },
            {
                "record": "s30761",
                "lead_number": 0,
                "urd_intervals": [
                    {"start_index": 4623000, "end_index": 4806500},
                    {"start_index": 7603000, "end_index": 7655000},
                    {"start_index": 7840500, "end_index": 7939000},
                    {"start_index": 15088000, "end_index": 15418500}
                ]
            },
            {
                "record": "s30761",
                "lead_number": 1,
                "urd_intervals": [
                    {"start_index": 4623000, "end_index": 4806500},
                    {"start_index": 7603000, "end_index": 7655000},
                    {"start_index": 7840500, "end_index": 7939000},
                    {"start_index": 15088000, "end_index": 15418500}
                ]
            },
            {
                "record": "s30761",
                "lead_number": 2,
                "urd_intervals": [
                    {"start_index": 4623000, "end_index": 4806500},
                    {"start_index": 7603000, "end_index": 7655000},
                    {"start_index": 7840500, "end_index": 7939000},
                    {"start_index": 15088000, "end_index": 15418500}
                ]
            },
            {
                "record": "s30801",
                "lead_number": 0,
                "urd_intervals": [
                    {"start_index": 678250, "end_index": 748750},
                    {"start_index": 926250, "end_index": 969750},
                    {"start_index": 4628750, "end_index": 4848250},
                    {"start_index": 5502750, "end_index": 5712250},
                    {"start_index": 6264250, "end_index": 6593750},
                    {"start_index": 7140750, "end_index": 7445750},
                    {"start_index": 9173750, "end_index": 9354750},
                    {"start_index": 10019750, "end_index": 10287750},
                    {"start_index": 12691250, "end_index": 12749250},
                    {"start_index": 14643250, "end_index": 14723750},
                    {"start_index": 14920750, "end_index": 15083750},
                    {"start_index": 16280250, "end_index": 16342750},
                    {"start_index": 16439750, "end_index": 16740250},
                    {"start_index": 17252750, "end_index": 17449750},
                    {"start_index": 17830250, "end_index": 18069250},
                    {"start_index": 18288750, "end_index": 20759750},
                    {"start_index": 20927250, "end_index": 21031250}
                ]
            },
            {
                "record": "s30801",
                "lead_number": 1,
                "urd_intervals": [
                    {"start_index": 17962250, "end_index": 18034250},
                    {"start_index": 20424750, "end_index": 20761250}
                ]
            },
            {
                "record": "s30801",
                "lead_number": 2,
                "urd_intervals": [
                    {"start_index": 17960250, "end_index": 18033250},
                    {"start_index": 20425750, "end_index": 20764250}
                ]
            }
        ]
        """
        all unreadable intervals are extracted from *_sta.json (stb, stc the same results) files
        using the following code

        with_urd = []
        for rec in wfdb.get_record_list("ltstdb"):
            with open(db_dir+rec+"_sta.json") as data_file:
                annos = json.load(data_file)
            for k, v in annos.items():
                nb_urd_intervals = len(v["urd_intervals"])
                if nb_urd_intervals > 0:
                    print("record", rec, "has", nb_urd_intervals, "unreadable intervals", "in lead", v["lead_number"], "as follows")
                    print(*v["urd_intervals"],sep="\n")
                    with_urd.append({
                        "record": rec,
                        "lead_number": v["lead_number"],
                        "urd_intervals": v["urd_intervals"]
                    })
                else:
                    print("record", rec, "has no unreadable intervals in lead", v["lead_number"])
        """
        # the following lambda functions are used to 
        # separate different types of annotations in .st* files
        # self.st_eps_conditions = lambda note: ("st" in note) and ("+" in note or "-" in note)
        self.st_eps_conditions = lambda note: ("+" in note or "-" in note) and ("LRST" not in note)
        self.significant_shift_conditions = lambda note: ("st" in note) and ("+" not in note) and ("-" not in note)  # can be simplied if put after self.st_eps_conditions
        self.global_ref_conditions = lambda note: "GRST" in note
        self.local_ref_conditions = lambda note: "LRST" in note
        self.urd_ep_conditions = lambda note: "urd" in note

        self.st_eps_with_lead_conditions = lambda note, lead_number: "st"+str(lead_number)+"+" in note or "st"+str(lead_number)+"-" in note
        self.global_ref_with_lead_conditions = lambda note, lead_number: "GRST"+str(lead_number) in note
        self.local_ref_with_lead_conditions = lambda note, lead_number: "LRST"+str(lead_number) in note
        self.urd_eps_with_lead_conditions = lambda note, lead_number: "urd"+str(lead_number) in note


    def _ls_rec(self, local:bool=True) -> NoReturn:
        """ finished, checked,

        find all records (relative path without file extension),
        and save into `self._all_records` for further use

        Parameters:
        -----------
        local: bool, default True,
            if True, read from local storage, prior to using `wfdb.get_record_list`
        """
        try:
            super()._ls_rec(local=local)
        except:
            # the first digit in the record name (2 or 3) indicates the number of ECG signals
            # records obtained from the same subject have names that differ in the last digit only
            self._all_records = [
                "s20011", "s20021", "s20031", "s20041", "s20051", "s20061", "s20071",
                "s20081", "s20091", "s20101", "s20111", "s20121", "s20131", "s20141",
                "s20151", "s20161", "s20171", "s20181", "s20191", "s20201", "s20211",
                "s20221", "s20231", "s20241", "s20251", "s20261", "s20271", "s20272",
                "s20273", "s20274", "s20281", "s20291", "s20301", "s20311", "s20321",
                "s20331", "s20341", "s20351", "s20361", "s20371", "s20381", "s20391",
                "s20401", "s20411", "s20421", "s20431", "s20441", "s20451", "s20461",
                "s20471", "s20481", "s20491", "s20501", "s20511", "s20521", "s20531",
                "s20541", "s20551", "s20561", "s20571", "s20581", "s20591", "s20601",
                "s20611", "s20621", "s20631", "s20641", "s20651", "s30661", "s30671",
                "s30681", "s30691", "s30701", "s30711", "s30721", "s30731", "s30732",
                "s30741", "s30742", "s30751", "s30752", "s30761", "s30771", "s30781",
                "s30791", "s30801",
            ]


    def get_subject_id(self, rec) -> int:
        """

        """
        raise NotImplementedError


    def database_info(self) -> NoReturn:
        """

        """
        print(self.__doc__)
