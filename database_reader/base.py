# -*- coding: utf-8 -*-
"""
Base classes for datasets from different sources:
    Physionet
    NSRR
    Image datasets
    Other

Remarks:
1. for whole-dataset visualizing: http://zzz.bwh.harvard.edu/luna/vignettes/dataplots/
2. visualizing using UMAP: http://zzz.bwh.harvard.edu/luna/vignettes/nsrr-umap/
"""
import os
import sys
import pprint
import logging
import time
import json
from collections import namedtuple
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

import wfdb
import numpy as np
import pandas as pd
from pyedflib import EdfReader

from .utils.common import *


__all__ = [
    "PhysioNetDataBase",
    "NSRRDataBase",
    "ImageDataBase",
    "AudioDataBase",
    "OtherDataBase",
    "ECGWaveForm",
]


class _DataBase(object):
    """

    universal base class for all databases
    """
    def __init__(self, db_name:str, db_dir:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_name: str,
            name of the database
        db_dir: str, optional,
            storage path of the database,
            if not specified, `wfdb` will fetch data from the website of PhysioNet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        kwargs: dict,
        """
        self.db_name = db_name
        self.db_dir = db_dir
        self.working_dir = working_dir or os.getcwd()
        os.makedirs(self.working_dir, exist_ok=True)
        self.data_ext = None
        self.ann_ext = None
        self.header_ext = "hea"
        self.verbose = verbose
        self.logger = None
        self._set_logger(prefix=type(self).__name__)

    def _ls_rec(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    def _set_logger(self, prefix:Optional[str]=None) -> NoReturn:
        """

        Parameters:
        -----------
        prefix: str, optional,
            prefix (for each line) of the logger, and its file name
        """
        _prefix = prefix+"-" if prefix else ""
        self.logger = logging.getLogger('{}-{}-logger'.format(_prefix, self.db_name))
        log_filepath = os.path.join(self.working_dir, "{}{}.log".format(_prefix, self.db_name))
        print("log file path is set {}".format(log_filepath))

        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_filepath)
        if self.verbose >= 2:
            print("levels of c_handler and f_handler are set DEBUG")
            c_handler.setLevel(logging.DEBUG)
            f_handler.setLevel(logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
        elif self.verbose >= 1:
            print("level of c_handler is set INFO, level of f_handler is set DEBUG")
            c_handler.setLevel(logging.INFO)
            f_handler.setLevel(logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
        else:
            print("levels of c_handler and f_handler are set WARNING")
            c_handler.setLevel(logging.WARNING)
            f_handler.setLevel(logging.WARNING)
            self.logger.setLevel(logging.WARNING)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def train_test_split(self):
        """
        """
        raise NotImplementedError


class PhysioNetDataBase(_DataBase):
    """
    https://www.physionet.org/
    """
    def __init__(self, db_name:str, db_dir:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_name: str,
            name of the database
        db_dir: str, optional,
            storage path of the database,
            if not specified, `wfdb` will fetch data from the website of PhysioNet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        kwargs: dict,

        NOTE:
        -----

        typical `db_dir`:
        ------------------
            "E:\\notebook_dir\\ecg\\data\\PhysioNet\\xxx\\"
            "/export/algo/wenh06/ecg_data/xxx/"
        """
        super().__init__(db_name=db_name, db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        # self._set_logger(prefix="PhysioNet")
        """
        `self.freq` for those with single signal source, e.g. ECG,
        for those with multiple signal sources like PSG, self.freq is default to the frequency of ECG if ECG applicable
        """
        self.freq = None
        self.all_records = None

        self.wfdb_rec = None
        self.wfdb_ann = None
        self.device_id = None  # maybe data are imported into impala db, to facilitate analyzing

        if self.verbose <= 2:
            self.df_all_db_info = pd.DataFrame()
            return
        
        try:
            all_dbs = wfdb.io.get_dbs()
        except:
            all_dbs = [
                ['adfecgdb', 'Abdominal and Direct Fetal ECG Database'],
                ['aftdb', 'AF Termination Challenge Database'],
                ['ahadb', 'AHA Database [sample excluded record]'],
                ['aami-ec13', 'ANSI/AAMI EC13 Test Waveforms'],
                ['apnea-ecg', 'Apnea-ECG Database'],
                ['chfdb', 'BIDMC Congestive Heart Failure Database'],
                ['bidmc', 'BIDMC PPG and Respiration Dataset'],
                ['bpssrat', 'Blood Pressure in Salt-Sensitive Dahl Rats'],
                ['capslpdb', 'CAP Sleep Database'],
                ['crisdb', 'CAST RR Interval Sub-Study Database'],
                ['cves', 'Cerebral Vasoregulation in Elderly with Stroke'],
                ['challenge/2009/test-set-a', 'Challenge 2009 Test Set A'],
                ['challenge/2009/test-set-b', 'Challenge 2009 Test Set B'],
                ['challenge/2010/set-a', 'Challenge 2010 Training Set A'],
                ['challenge/2010/set-b', 'Challenge 2010 Test Set B'],
                ['challenge/2010/set-c', 'Challenge 2010 Test Set C'],
                ['challenge/2011/sim', 'Challenge 2011 Pilot Set'],
                ['challenge/2011/set-a', 'Challenge 2011 Training Set A'],
                ['challenge/2011/set-b', 'Challenge 2011 Test Set B'],
                ['challenge/2013/set-a', 'Challenge 2013 Training Set A'],
                ['challenge/2013/set-b', 'Challenge 2013 Test Set B'],
                ['challenge/2014/set-p2', 'Challenge 2014 Extended Training Set'],
                ['challenge/2014/set-p', 'Challenge 2014 Training Set'],
                ['challenge/2015/training', 'Challenge 2015 Training Set'],
                ['challenge/2016/training-a', 'Challenge 2016 Training Set A'],
                ['challenge/2016/training-b', 'Challenge 2016 Training Set B'],
                ['challenge/2016/training-c', 'Challenge 2016 Training Set C'],
                ['challenge/2016/training-d', 'Challenge 2016 Training Set D'],
                ['challenge/2016/training-e', 'Challenge 2016 Training Set E'],
                ['challenge/2016/training-f', 'Challenge 2016 Training Set F'],
                ['challenge/2017/training', 'Challenge 2017 Training Set'],
                ['challenge/2018/training', 'Challenge 2018 Training Set'],
                ['challenge/2018/test', 'Challenge 2018 Test Set'],
                ['charisdb', 'CHARIS database'],
                ['chbmit', 'CHB-MIT Scalp EEG Database'],
                ['cebsdb', 'Combined measurement of ECG, Breathing and Seismocardiograms'],
                ['culm', 'Complex Upper-Limb Movements'],
                ['chf2db', 'Congestive Heart Failure RR Interval Database'],
                ['ctu-uhb-ctgdb', 'CTU-CHB Intrapartum Cardiotocography Database'],
                ['cudb', 'CU Ventricular Tachyarrhythmia Database'],
                ['ecgdmmld',
                'ECG Effects of Dofetilide, Moxifloxacin, Dofetilide+Mexiletine, Dofetilide+Lidocaine and Moxifloxacin+Diltiazem'],
                ['ecgcipa', 'CiPA ECG Validation Study'],
                ['ecgrdvq',
                'ECG Effects of Ranolazine, Dofetilide, Verapamil, and Quinidine'],
                ['ecgiddb', 'ECG-ID Database'],
                ['eegmat', 'EEG During Mental Arithmetic Tasks'],
                ['eegmmidb', 'EEG Motor Movement/Imagery Dataset'],
                ['ltrsvp', 'EEG Signals from an RSVP Task'],
                ['erpbci', 'ERP-based Brain-Computer Interface recordings'],
                ['edb', 'European ST-T Database'],
                ['earh', 'Evoked Auditory Responses in Heading Impaired'],
                ['earndb', 'Evoked Auditory Responses in Normals'],
                ['emgdb', 'Examples of Electromyograms'],
                ['fantasia', 'Fantasia Database'],
                ['fecgsyndb', 'Fetal ECG Synthetic Database'],
                ['fpcgdb', 'Fetal PCG Database'],
                ['gaitdb', 'Gait in Aging and Disease Database'],
                ['gaitndd', 'Gait in Neurodegenerative Disease Database'],
                ['gait-maturation-db/data', 'Gait Maturation Database'],
                ['meditation/data', 'Heart Rate Oscillations during Meditation'],
                ['hbedb', 'Human Balance Evaluation Database'],
                ['ehgdb', 'Icelandic 16-electrode Electrohysterogram Database'],
                ['iafdb', 'Intracardiac Atrial Fibrillation Database'],
                ['ltafdb', 'Long Term AF Database'],
                ['ltstdb', 'Long Term ST Database'],
                ['mssvepdb', 'MAMEM SSVEP Database'],
                ['mghdb', 'MGH/MF Waveform Database'],
                ['mimicdb', 'MIMIC Database'],
                ['mimicdb/numerics', 'MIMIC Database Numerics'],
                ['mimic2cdb', 'MIMIC II Clinical Database Public Subset'],
                ['mimic2wdb/30', 'MIMIC II/III Waveform Database, part 0'],
                ['mimic2wdb/31', 'MIMIC II/III Waveform Database, part 1'],
                ['mimic2wdb/32', 'MIMIC II/III Waveform Database, part 2'],
                ['mimic2wdb/33', 'MIMIC II/III Waveform Database, part 3'],
                ['mimic2wdb/34', 'MIMIC II/III Waveform Database, part 4'],
                ['mimic2wdb/35', 'MIMIC II/III Waveform Database, part 5'],
                ['mimic2wdb/36', 'MIMIC II/III Waveform Database, part 6'],
                ['mimic2wdb/37', 'MIMIC II/III Waveform Database, part 7'],
                ['mimic2wdb/38', 'MIMIC II/III Waveform Database, part 8'],
                ['mimic2wdb/39', 'MIMIC II/III Waveform Database, part 9'],
                ['mimic2wdb/matched', 'MIMIC II Waveform Database Matched Subset'],
                ['mimic3wdb/matched', 'MIMIC III Waveform Database Matched Subset'],
                ['mitdb', 'MIT-BIH Arrhythmia Database'],
                ['pwave', 'MIT-BIH Arrhythmia Database P-Wave Annotations'],
                ['afdb', 'MIT-BIH Atrial Fibrillation Database'],
                ['cdb', 'MIT-BIH ECG Compression Test Database'],
                ['ltdb', 'MIT-BIH Long-Term ECG Database'],
                ['vfdb', 'MIT-BIH Malignant Ventricular Ectopy Database'],
                ['nstdb', 'MIT-BIH Noise Stress Test Database'],
                ['nsrdb', 'MIT-BIH Normal Sinus Rhythm Database'],
                ['excluded', '... [Recordings excluded from the NSR DB]'],
                ['slpdb', 'MIT-BIH Polysomnographic Database'],
                ['stdb', 'MIT-BIH ST Change Database'],
                ['svdb', 'MIT-BIH Supraventricular Arrhythmia Database'],
                ['mmgdb', 'MMG Database'],
                ['macecgdb', 'Motion Artifact Contaminated ECG Database'],
                ['motion-artifact', 'Motion Artifact Contaminated fNIRS and EEG Data'],
                ['noneeg', 'Non-EEG Dataset for Assessment of Neurological Status'],
                ['nifecgdb', 'Non-Invasive Fetal ECG Database'],
                ['nifeadb', 'Non-Invasive Fetal ECG Arrhythmia Database'],
                ['nsr2db', 'Normal Sinus Rhythm RR Interval Database'],
                ['ob1db', 'OB-1 Fetal ECG Database [sample record]'],
                ['afpdb', 'PAF Prediction Challenge Database'],
                ['osv', 'Pattern Analysis of Oxygen Saturation Variability'],
                ['prcp', 'Physiologic Response to Changes in Posture'],
                ['szdb', 'Post-Ictal Heart Rate Oscillations in Partial Epilepsy'],
                ['picsdb', 'Preterm Infant Cardio-Respiratory Signals Database'],
                ['ptbdb', 'PTB Diagnostic ECG Database'],
                ['qtdb', 'QT Database'],
                ['rvmh1', 'Response to Valsalva Maneuver in Humans'],
                ['sufhsdb', 'Shiraz University Fetal Heart Sounds Database'],
                ['simfpcgdb', 'Simulated Fetal Phonocardiograms'],
                ['sleepbrl', 'Sleep Bioradiolocation Database'],
                ['sleep-edfx', 'Sleep-EDF Database [Expanded]'],
                ['shhpsgdb', 'Sleep Heart Health Study PSG Database [sample record]'],
                ['shareedb',
                'Smart Health for Assessing the Risk of Events via ECG Database'],
                ['mvtdb/data', 'Spontaneous Ventricular Tachyarrhythmia Database'],
                ['sgamp', 'Squid Giant Axon Membrane Potential'],
                ['incartdb', 'St Petersburg INCART 12-lead Arrhythmia Database'],
                ['staffiii', 'STAFF III Database'],
                ['drivedb', 'Stress Recognition in Automobile Drivers'],
                ['sddb', 'Sudden Cardiac Death Holter Database'],
                ['twadb', 'T-Wave Alternans Challenge Database'],
                ['taichidb', 'Tai Chi, Physiological Complexity, and Healthy Aging - Gait'],
                ['tpehgdb', 'Term-Preterm EHG Database'],
                ['tpehgt', 'Term-Preterm EHG DataSet with Tocogram (TPEHGT DS)'],
                ['ucddb', 'UCD Sleep Apnea Database'],
                ['unicaprop', 'UniCA ElectroTastegram Database (PROP)'],
                ['videopulse', 'Video Pulse Signals in Stationary and Motion Conditions'],
                ['voiced', 'VOice ICar fEDerico II Database'],
                ['wrist', 'Wrist PPG During Exercise'],
                ['mimic2db', 'MIMIC II Waveform DB, v2 [deprecated, use v3]'],
                ['mimic2db/numerics',
                'MIMIC II Waveform DB, v2 Numerics [deprecated, use v3]'],
                ['sleep-edf', 'Sleep-EDF Database, v1 [deprecated, use sleep-edfx]']
            ]
        self.df_all_db_info = pd.DataFrame(
            {
                "db_name": [item[0] for item in all_dbs],
                "db_description": [item[1] for item in all_dbs]
            }
        )

    
    def _ls_rec(self, db_name:Optional[str]=None, local:bool=True) -> NoReturn:
        """ finished, checked,

        find all records (relative path without file extension),
        and save into `self.all_records` for further use

        Parameters:
        -----------
        db_name: str, optional,
            name of the database for using `wfdb.get_record_list`,
            if not set, `self.db_name` will be used
        local: bool, default True,
            if True, read from local storage, prior to using `wfdb.get_record_list`
        """
        if local:
            self._ls_rec_local()
            return
        try:
            self.all_records = wfdb.get_record_list(db_name or self.db_name)
        except:
            self._ls_rec_local()
            

    def _ls_rec_local(self,) -> NoReturn:
        """ finished, checked,

        find all records in `self.db_dir`
        """
        record_list_fp = os.path.join(self.db_dir, "record_list.json")
        if os.path.isfile(record_list_fp):
            with open(record_list_fp, "r") as f:
                self.all_records = json.load(f)
        else:
            print("Please wait patiently to let the reader find all records of all the tranches...")
            start = time.time()
            self.all_records = get_record_list_recursive(self.db_dir, self.data_ext)
            print(f"Done in {time.time() - start} seconds!")
            with open(record_list_fp, "w") as f:
                json.dump(self.all_records, f)


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
        

    def get_patient_id(self, rec:str) -> int:
        """
        synonym for func `self.get_subject_id`
        """
        return self.get_subject_id(rec=rec)


    def load_data(self, rec:str, **kwargs) -> Any:
        """
        load data from the record `rec`
        """
        raise NotImplementedError


    def load_ecg_data(self, rec:str, **kwargs) -> np.ndarray:
        """
        load ECG data from the record `rec`
        """
        raise NotImplementedError


    def load_ann(self, rec:str, **kwargs) -> Any:
        """
        load annotations of the record `rec`

        NOTE that the records might have several annotation files
        """
        raise NotImplementedError


    def database_info(self, detailed:bool=False) -> NoReturn:
        """
        print the information about the database

        detailed: bool, default False,
            if False, an short introduction of the database will be printed,
            if True, then docstring of the class will be printed additionally
        """
        raise NotImplementedError


    def helper(self, items:Union[List[str],str,type(None)]=None, **kwargs) -> NoReturn:
        """ finished, checked, to be improved,

        print corr. meanings of symbols belonging to `items`

        Parameters:
        items: str, or list of str, optional,
            the items to print,
            if not specified, then a comprehensive printing of meanings of all symbols will be performed
        """
        attrs = vars(self)
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith('__') and func.endswith('__'))]

        beat_annotations = {
            'N': 'Normal beat',
            'L': 'Left bundle branch block beat',
            'R': 'Right bundle branch block beat',
            'B': 'Bundle branch block beat (unspecified)',
            'A': 'Atrial premature beat',
            'a': 'Aberrated atrial premature beat',
            'J': 'Nodal (junctional) premature beat',
            'S': 'Supraventricular premature or ectopic beat (atrial or nodal)',
            'V': 'Premature ventricular contraction',
            'r': 'R-on-T premature ventricular contraction',
            'F': 'Fusion of ventricular and normal beat',
            'e': 'Atrial escape beat',
            'j': 'Nodal (junctional) escape beat',
            'n': 'Supraventricular escape beat (atrial or nodal)',
            'E': 'Ventricular escape beat',
            '/': 'Paced beat',
            'f': 'Fusion of paced and normal beat',
            'Q': 'Unclassifiable beat',
            '?': 'Beat not classified during learning'
        }

        non_beat_annotations = {
            '[': 'Start of ventricular flutter/fibrillation',
            '!': 'Ventricular flutter wave',
            ']': 'End of ventricular flutter/fibrillation',
            'x': 'Non-conducted P-wave (blocked APC)',
            '(': 'Waveform onset',
            ')': 'Waveform end',
            'p': 'Peak of P-wave',
            't': 'Peak of T-wave',
            'u': 'Peak of U-wave',
            '`': 'PQ junction',
            "'": 'J-point',
            '^': '(Non-captured) pacemaker artifact',
            '|': 'Isolated QRS-like artifact',
            '~': 'Change in signal quality',
            '+': 'Rhythm change',
            's': 'ST segment change',
            'T': 'T-wave change',
            '*': 'Systole',
            'D': 'Diastole',
            '=': 'Measurement annotation',
            '"': 'Comment annotation',
            '@': 'Link to external data'
        }

        rhythm_annotations = {
            '(AB': 'Atrial bigeminy',
            '(AFIB': 'Atrial fibrillation',
            '(AFL': 'Atrial flutter',
            '(B': 'Ventricular bigeminy',
            '(BII': '2° heart block',
            '(IVR': 'Idioventricular rhythm',
            '(N': 'Normal sinus rhythm',
            '(NOD': 'Nodal (A-V junctional) rhythm',
            '(P': 'Paced rhythm',
            '(PREX': 'Pre-excitation (WPW)',
            '(SBR': 'Sinus bradycardia',
            '(SVTA': 'Supraventricular tachyarrhythmia',
            '(T': 'Ventricular trigeminy',
            '(VFL': 'Ventricular flutter',
            '(VT': 'Ventricular tachycardia'
        }

        all_annotations = [
            beat_annotations,
            non_beat_annotations,
            rhythm_annotations
        ]

        summary_items = [
            "beat",
            "non-beat",
            'rhythm'
        ]

        if items is None:
            _items = ['attributes', 'methods', 'beat', 'non-beat', 'rhythm']
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if 'attributes' in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if 'methods' in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)
        if 'beat' in _items:
            print("--- helpler - beat ---")
            pp.pprint(beat_annotations)
        if 'non-beat' in _items:
            print("--- helpler - non-beat ---")
            pp.pprint(non_beat_annotations)
        if 'rhythm' in _items:
            print("--- helpler - rhythm ---")
            pp.pprint(rhythm_annotations)

        for k in _items:
            if k in summary_items:
                continue
            for a in all_annotations:
                if k in a.keys() or '('+k in a.keys():
                    try:
                        print("{0} stands for {1}".format(k.split('(')[1], a[k]))
                    except:
                        print("{0} stands for {1}".format(k, a['('+k]))


class NSRRDataBase(_DataBase):
    """
    https://sleepdata.org/
    """
    def __init__(self, db_name:str, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_name: str,
            name of the database
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        kwargs: dict,

        typical `db_dir`:
        ------------------
            "E:\\notebook_dir\\ecg\\data\\NSRR\\xxx\\"
            "/export/algo/wenh06/ecg_data/NSRR/xxx/"
        """
        super().__init__(db_name=db_name, db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        # self._set_logger(prefix="NSRR")
        self.freq = None
        self.all_records = None
        self.device_id = None  # maybe data are imported into impala db, to facilitate analyzing
        self.file_opened = None
        
        all_dbs = [
            ["shhs", "Multi-cohort study focused on sleep-disordered breathing and cardiovascular outcomes"],
            ["mesa", ""],
            ["oya", ""],
            ["chat", "Multi-center randomized trial comparing early adenotonsillectomy to watchful waiting plus supportive care"],
            ["heartbeat", "Multi-center Phase II randomized controlled trial that evaluates the effects of supplemental nocturnal oxygen or Positive Airway Pressure (PAP) therapy"],
            # more to be added
        ]
        self.df_all_db_info = pd.DataFrame(
            {
                "db_name": [item[0] for item in all_dbs],
                "db_description": [item[1] for item in all_dbs]
            }
        )
        self.kwargs = kwargs


    def safe_edf_file_operation(self, operation:str='close', full_file_path:Optional[str]=None) -> Union[EdfReader, NoReturn]:
        """ finished, checked,

        Parameters:
        -----------
        operation: str, default 'close',
            operation name, can be 'open' and 'close'
        full_file_path: str, optional,
            path of the file which contains the psg data,
            if not given, default path will be used
        
        Returns:
        --------

        """
        if operation == 'open':
            if self.file_opened is not None:
                self.file_opened._close()
            self.file_opened = EdfReader(full_file_path)
        elif operation =='close':
            if self.file_opened is not None:
                self.file_opened._close()
                self.file_opened = None
        else:
            raise ValueError("Illegal operation")
        

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
        

    def get_patient_id(self, rec:str) -> int:
        """
        synonym for func `self.get_subject_id`
        """
        return self.get_subject_id(rec=rec)


    def show_rec_stats(self, rec:str) -> NoReturn:
        """
        print the statistics about the record `rec`

        Parameters:
        -----------
        rec: str,
            record name
        """
        raise NotImplementedError


    def database_info(self, detailed:bool=False) -> NoReturn:
        """
        print the information about the database

        detailed: bool, default False,
            if False, an short introduction of the database will be printed,
            if True, then docstring of the class will be printed additionally
        """
        raw_info = {
            "What": "",
            "Who": "",
            "When": "",
            "Funding": ""
        }

        print(raw_info)
        
        if detailed:
            print(self.__doc__)


    def helper(self, items:Union[List[str],str,type(None)]=None, **kwargs) -> NoReturn:
        """
        """
        pp = pprint.PrettyPrinter(indent=4)

        attrs = vars(self)
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith('__') and func.endswith('__'))]

        if items is None:
            _items = ['attributes', 'methods', ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if 'attributes' in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if 'methods' in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)


class ImageDataBase(_DataBase):
    """

    """
    def __init__(self, db_name:str, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_name: str,
            name of the database
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        kwargs: dict,
        """
        super().__init__(db_name=db_name, db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        # self._set_logger(prefix=None)


    def helper(self, items:Union[List[str],str,type(None)]=None, **kwargs) -> NoReturn:
        """
        """
        pp = pprint.PrettyPrinter(indent=4)

        attrs = vars(self)
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith('__') and func.endswith('__'))]

        if items is None:
            _items = ['attributes', 'methods', ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if 'attributes' in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if 'methods' in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)


class AudioDataBase(object):
    """

    """
    def __init__(self, db_name:str, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_name: str,
            name of the database
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        kwargs: dict,
        """
        super().__init__(db_name=db_name, db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        # self._set_logger(prefix=None)


    def helper(self, items:Union[List[str],str,type(None)]=None, **kwargs) -> NoReturn:
        """
        """
        pp = pprint.PrettyPrinter(indent=4)

        attrs = vars(self)
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith('__') and func.endswith('__'))]

        if items is None:
            _items = ['attributes', 'methods', ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if 'attributes' in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if 'methods' in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)


class OtherDataBase(object):
    """

    """
    def __init__(self, db_name:str, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        r"""
        Parameters:
        -----------
        db_name: str,
            name of the database
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        kwargs: dict,

        typical 'db_dir':
        ------------------
            "E:\\notebook_dir\\ecg\\data\xxx\\"
            "/export/algo/wenh06/ecg_data/xxx/"
        """
        super().__init__(db_name=db_name, db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        # self._set_logger(prefix=None)

        self.freq = None
        self.all_records = None
        self.device_id = None  # maybe data are imported into impala db, to facilitate analyzing
        
        self.kwargs = kwargs
        

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
        

    def get_patient_id(self, rec:str) -> int:
        """
        synonym for func `self.get_subject_id`
        """
        return self.get_subject_id(rec=rec)
    

    def database_info(self, detailed:bool=False) -> NoReturn:
        """
        print the information about the database

        detailed: bool, default False,
            if False, an short introduction of the database will be printed,
            if True, then docstring of the class will be printed additionally
        """
        raw_info = {}

        print(raw_info)
        
        if detailed:
            print(self.__doc__)


    def helper(self, items:Union[List[str],str,type(None)]=None, **kwargs) -> NoReturn:
        """
        """
        pp = pprint.PrettyPrinter(indent=4)
        
        attrs = vars(self)
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith('__') and func.endswith('__'))]

        if items is None:
            _items = ['attributes', 'methods', ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if 'attributes' in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if 'methods' in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)



ECGWaveForm = namedtuple(
    typename='ECGWaveForm',
    field_names=['name', 'onset', 'offset', 'peak', 'duration'],
)