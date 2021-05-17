# -*- coding: utf-8 -*-
"""
Base classes for datasets from different sources:
    Physionet
    NSRR
    Image databases
    Audio databases
    Other databases

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
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
from pyedflib import EdfReader

from .utils.common import *


__all__ = [
    "WFDB_Beat_Annotations", "WFDB_Non_Beat_Annotations", "WFDB_Rhythm_Annotations",
    "PhysioNetDataBase",
    "NSRRDataBase",
    "ImageDataBase",
    "AudioDataBase",
    "OtherDataBase",
    "ECGWaveForm",
]


WFDB_Beat_Annotations = {
    "N": "Normal beat",
    "L": "Left bundle branch block beat",
    "R": "Right bundle branch block beat",
    "B": "Bundle branch block beat (unspecified)",
    "A": "Atrial premature beat",
    "a": "Aberrated atrial premature beat",
    "J": "Nodal (junctional) premature beat",
    "S": "Supraventricular premature or ectopic beat (atrial or nodal)",
    "V": "Premature ventricular contraction",
    "r": "R-on-T premature ventricular contraction",
    "F": "Fusion of ventricular and normal beat",
    "e": "Atrial escape beat",
    "j": "Nodal (junctional) escape beat",
    "n": "Supraventricular escape beat (atrial or nodal)",
    "E": "Ventricular escape beat",
    "/": "Paced beat",
    "f": "Fusion of paced and normal beat",
    "Q": "Unclassifiable beat",
    "?": "Beat not classified during learning",
}

WFDB_Non_Beat_Annotations = {
    "[": "Start of ventricular flutter/fibrillation",
    "!": "Ventricular flutter wave",
    "]": "End of ventricular flutter/fibrillation",
    "x": "Non-conducted P-wave (blocked APC)",
    "(": "Waveform onset",
    ")": "Waveform end",
    "p": "Peak of P-wave",
    "t": "Peak of T-wave",
    "u": "Peak of U-wave",
    "`": "PQ junction",
    "'": "J-point",
    "^": "(Non-captured) pacemaker artifact",
    "|": "Isolated QRS-like artifact",
    "~": "Change in signal quality",
    "+": "Rhythm change",
    "s": "ST segment change",
    "T": "T-wave change",
    "*": "Systole",
    "D": "Diastole",
    "=": "Measurement annotation",
    '"': "Comment annotation",
    "@": "Link to external data",
}

WFDB_Rhythm_Annotations = {
    "(AB": "Atrial bigeminy",
    "(AFIB": "Atrial fibrillation",
    "(AFL": "Atrial flutter",
    "(B": "Ventricular bigeminy",
    "(BII": "2° heart block",
    "(IVR": "Idioventricular rhythm",
    "(N": "Normal sinus rhythm",
    "(NOD": "Nodal (A-V junctional) rhythm",
    "(P": "Paced rhythm",
    "(PREX": "Pre-excitation (WPW)",
    "(SBR": "Sinus bradycardia",
    "(SVTA": "Supraventricular tachyarrhythmia",
    "(T": "Ventricular trigeminy",
    "(VFL": "Ventricular flutter",
    "(VT": "Ventricular tachycardia",
}


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
        self._all_records = None
        self._set_logger(prefix=type(self).__name__)

    def _ls_rec(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    def _auto_infer_units(self, sig:np.ndarray, sig_type:str="ECG") -> str:
        """ finished, checked,

        automatically infer the units of `sig`,
        under the assumption that `sig` not being raw signal, with baseline removed

        Parameters:
        -----------
        sig: ndarray,
            the signal to infer its units
        sig_type: str, default "ECG", case insensitive,
            type of the signal

        Returns:
        --------
        units: str,
            units of `sig`, "μV" or "mV"
        """
        if sig_type.lower() == "ecg":
            _MAX_mV = 20  # 20mV, seldom an ECG device has range larger than this value
            max_val = np.max(np.abs(sig))
            if max_val > _MAX_mV:
                units = "μV"
            else:
                units = "mV"
        else:
            raise NotImplementedError(f"not implemented for {sig_type}")
        return units

    def _set_logger(self, prefix:Optional[str]=None) -> NoReturn:
        """

        Parameters:
        -----------
        prefix: str, optional,
            prefix (for each line) of the logger, and its file name
        """
        _prefix = prefix+"-" if prefix else ""
        self.logger = logging.getLogger(f"{_prefix}-{self.db_name}-logger")
        log_filepath = os.path.join(self.working_dir, f"{_prefix}{self.db_name}.log")
        print(f"log file path is set \042{log_filepath}\042")

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
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    @property
    def all_records(self):
        """
        """
        if self._all_records is None:
            self._ls_rec()
        return self._all_records

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
        `self.fs` for those with single signal source, e.g. ECG,
        for those with multiple signal sources like PSG, self.fs is default to the frequency of ECG if ECG applicable
        """
        self.fs = None
        self._all_records = None

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
                ["aami-ec13", "ANSI/AAMI EC13 Test Waveforms"],
                ["adfecgdb", "Abdominal and Direct Fetal ECG Database"],
                ["afdb", "MIT-BIH Atrial Fibrillation Database"],
                ["afpdb", "PAF Prediction Challenge Database"],
                ["aftdb", "AF Termination Challenge Database"],
                ["ahadb", "AHA Database Sample Excluded Record"],
                ["antimicrobial-resistance-uti", "AMR-UTI: Antimicrobial Resistance in Urinary Tract Infections"],
                ["apnea-ecg", "Apnea-ECG Database"],
                ["bhx-brain-bounding-box", "Brain Hemorrhage Extended (BHX): Bounding box extrapolation from thick to thin slice CT images"],
                ["bidmc", "BIDMC PPG and Respiration Dataset"],
                ["bpssrat", "Blood Pressure in Salt-Sensitive Dahl Rats"],
                ["butqdb", "Brno University of Technology ECG Quality Database (BUT QDB)"], ["capslpdb", "CAP Sleep Database"],
                ["cdb", "MIT-BIH ECG Compression Test Database"],
                ["cded", "Cerebromicrovascular Disease in Elderly with Diabetes"],
                ["cebsdb", "Combined measurement of ECG, Breathing and Seismocardiograms"],
                ["cerebral-vasoreg-diabetes", "Cerebral Vasoregulation in Diabetes"],
                ["charisdb", "CHARIS database"],
                ["chbmit", "CHB-MIT Scalp EEG Database"],
                ["chf2db", "Congestive Heart Failure RR Interval Database"],
                ["chfdb", "BIDMC Congestive Heart Failure Database"],
                ["crisdb", "CAST RR Interval Sub-Study Database"],
                ["ct-ich", "Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation"],
                ["ct-ich", "Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation"],
                ["ct-ich", "Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation"],
                ["ct-ich", "Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation"],
                ["ctu-uhb-ctgdb", "CTU-CHB Intrapartum Cardiotocography Database"],
                ["cudb", "CU Ventricular Tachyarrhythmia Database"],
                ["cuiless16", "CUILESS2016"],
                ["culm", "Complex Upper-Limb Movements"],
                ["cves", "Cerebral Vasoregulation in Elderly with Stroke"],
                ["cxr-phone", "Smartphone-Captured Chest X-Ray Photographs"],
                ["deidentifiedmedicaltext", "Deidentified Medical Text"],
                ["drivedb", "Stress Recognition in Automobile Drivers"],
                ["earh", "Evoked Auditory Responses in Heading Impaired"],
                ["earndb", "Evoked Auditory Responses in Normals"],
                ["ecg-spider-clip", "Electrocardiogram, skin conductance and respiration from spider-fearful individuals watching spider video clips"],
                ["ecgcipa", "CiPA ECG Validation Study"],
                ["ecgdmmld", "ECG Effects of Dofetilide, Moxifloxacin, Dofetilide+Mexiletine, Dofetilide+Lidocaine and Moxifloxacin+Diltiazem"],
                ["ecgiddb", "ECG-ID Database"],
                ["ecgrdvq", "ECG Effects of Ranolazine, Dofetilide, Verapamil, and Quinidine"],
                ["edb", "European ST-T Database"],
                ["eegmat", "EEG During Mental Arithmetic Tasks"],
                ["eegmmidb", "EEG Motor Movement/Imagery Dataset"],
                ["egd-cxr", "Eye Gaze Data for Chest X-rays"],
                ["ehgdb", "Icelandic 16-electrode Electrohysterogram Database"],
                ["eicu-crd", "eICU Collaborative Research Database"],
                ["eicu-crd-demo", "eICU Collaborative Research Database Demo"],
                ["electrodermal-activity", "Electrodermal Activity of Healthy Volunteers while Awake and at Rest"],
                ["emer-complaint-gout", "Gout Emergency Department Chief Complaint Corpora"],
                ["emgdb", "Examples of Electromyograms"],
                ["erpbci", "ERP-based Brain-Computer Interface recordings"],
                ["excluded", "Recordings excluded from the NSR DB"],
                ["fantasia", "Fantasia Database"],
                ["fecgsyndb", "Fetal ECG Synthetic Database"],
                ["fpcgdb", "Fetal PCG Database"],
                ["gait-maturation-db", "Gait Maturation Database"],
                ["gaitdb", "Gait in Aging and Disease Database"],
                ["gaitndd", "Gait in Neurodegenerative Disease Database"],
                ["gaitpdb", "Gait in Parkinson's Disease"],
                ["hbedb", "Human Balance Evaluation Database"],
                ["heart-failure-zigong", "Hospitalized patients with heart failure: integrating electronic healthcare records and external outcome data"],
                ["heart-failure-zigong", "Hospitalized patients with heart failure: integrating electronic healthcare records and external outcome data"],
                ["hirid", "HiRID, a high time-resolution ICU dataset"],
                ["iafdb", "Intracardiac Atrial Fibrillation Database"],
                ["images", "Samples of MR Images"],
                ["incartdb", "St Petersburg INCART 12-lead Arrhythmia Database"],
                ["inipdmsa", "Safety and Preliminary Efficacy of Intranasal Insulin for Cognitive Impairment in Parkinson Disease and Multiple System Atrophy"],
                ["kinematic-actors-emotions", "Kinematic dataset of actors expressing emotions"],
                ["kinematic-actors-emotions", "Kinematic dataset of actors expressing emotions"],
                ["kinematic-actors-emotions", "Kinematic dataset of actors expressing emotions"],
                ["ltafdb", "Long Term AF Database"],
                ["ltdb", "MIT-BIH Long-Term ECG Database"],
                ["ltmm", "Long Term Movement Monitoring Database"],
                ["ltrsvp", "EEG Signals from an RSVP Task"],
                ["ltstdb", "Long Term ST Database"],
                ["ludb", "Lobachevsky University Electrocardiography Database"],
                ["macecgdb", "Motion Artifact Contaminated ECG Database"],
                ["maternal-visceral-adipose", "Visceral adipose tissue measurements during pregnancy"],
                ["meditation", "Heart Rate Oscillations during Meditation"],
                ["mednli", "MedNLI - A Natural Language Inference Dataset For The Clinical Domain"],
                ["mednli-bionlp19", "MedNLI for Shared Task at ACL BioNLP 2019"],
                ["mednli-bionlp19", "MedNLI for Shared Task at ACL BioNLP 2019"],
                ["mghdb", "MGH/MF Waveform Database"],
                ["mimic-cxr", "MIMIC-CXR Database"],
                ["mimic-cxr", "MIMIC-CXR Database"],
                ["mimic-cxr-jpg", "MIMIC-CXR-JPG - chest radiographs with structured labels"],
                ["mimic-seqex", "MIMIC-III - SequenceExamples for TensorFlow modeling"],
                ["mimic2-iaccd", "Clinical data from the MIMIC-II database for a case study on indwelling arterial catheters"],
                ["mimic3wdb", "MIMIC-III Waveform Database"],
                ["mimic3wdb-matched", "MIMIC-III Waveform Database Matched Subset"],
                ["mimicdb", "MIMIC Database"],
                ["mimiciii", "MIMIC-III Clinical Database"],
                ["mimiciii-demo", "MIMIC-III Clinical Database Demo"],
                ["mimiciv", "MIMIC-IV"],
                ["mimiciv", "MIMIC-IV"],
                ["mitdb", "MIT-BIH Arrhythmia Database"],
                ["mmash", "Multilevel Monitoring of Activity and Sleep in Healthy People"],
                ["mmgdb", "MMG Database"],
                ["motion-artifact", "Motion Artifact Contaminated fNIRS and EEG Data"],
                ["mssvepdb", "MAMEM SSVEP Database"],
                ["music-motion-2012", "MICRO Motion capture data from groups of participants standing still to auditory stimuli (2012)"],
                ["mvtdb", "Spontaneous Ventricular Tachyarrhythmia Database"],
                ["nesfdb", "Noise Enhancement of Sensorimotor Function"],
                ["nifeadb", "Non-Invasive Fetal ECG Arrhythmia Database"],
                ["nifecgdb", "Non-Invasive Fetal ECG Database"],
                ["ninfea", "NInFEA: Non-Invasive Multimodal Foetal ECG-Doppler Dataset for Antenatal Cardiology Research"],
                ["noneeg", "Non-EEG Dataset for Assessment of Neurological Status"],
                ["nqmitcsxpd", "neuroQWERTY MIT-CSXPD Dataset"],
                ["nsr2db", "Normal Sinus Rhythm RR Interval Database"],
                ["nsrdb", "MIT-BIH Normal Sinus Rhythm Database"],
                ["nstdb", "MIT-BIH Noise Stress Test Database"], 
                ["ob1db", "OB-1 Fetal ECG Database"],
                ["osv", "Pattern Analysis of Oxygen Saturation Variability"],
                ["phdsm", "Permittivity of Healthy and Diseased Skeletal Muscle"],
                ["phdsm", "Permittivity of Healthy and Diseased Skeletal Muscle"],
                ["phenotype-annotations-mimic", "Phenotype Annotations for Patient Notes in the MIMIC-III Database"],
                ["physiozoo", "PhysioZoo - mammalian NSR databases"],
                ["picdb", "Paediatric Intensive Care database"],
                ["picdb", "Paediatric Intensive Care database"],
                ["picsdb", "Preterm Infant Cardio-Respiratory Signals Database"],
                ["plantar", "Modulation of Plantar Pressure and Muscle During Gait"],
                ["pmd", "A Pressure Map Dataset for In-bed Posture Classification"],
                ["prcp", "Physiologic Response to Changes in Posture"],
                ["ptb-xl", "PTB-XL, a large publicly available electrocardiography dataset"],
                ["ptb-xl", "PTB-XL, a large publicly available electrocardiography dataset"],
                ["ptbdb", "PTB Diagnostic ECG Database"],
                ["pwave", "MIT-BIH Arrhythmia Database P-Wave Annotations"],
                ["qde", "Quantitative Dehydration Estimation"],
                ["qtdb", "QT Database"],
                ["rvmh1", "Response to Valsalva Maneuver in Humans"],
                ["santa-fe", "Santa Fe Time Series Competition Data Set B"],
                ["sddb", "Sudden Cardiac Death Holter Database"],
                ["sgamp", "Squid Giant Axon Membrane Potential"],
                ["shareedb", "Smart Health for Assessing the Risk of Events via ECG Database"],
                ["shhpsgdb", "Sleep Heart Health Study PSG Database"],
                ["siena-scalp-eeg", "Siena Scalp EEG Database"],
                ["simfpcgdb", "Simulated Fetal Phonocardiograms"],
                ["simultaneous-measurements", "Simultaneous physiological measurements with five devices at different cognitive and physical loads"],
                ["sleep-accel", "Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography"],
                ["sleep-edf", "Sleep-EDF Database"],
                ["sleep-edfx", "Sleep-EDF Database Expanded"],
                ["sleepbrl", "Sleep Bioradiolocation Database"],
                ["slpdb", "MIT-BIH Polysomnographic Database"],
                ["staffiii", "STAFF III Database"],
                ["stdb", "MIT-BIH ST Change Database"],
                ["sufhsdb", "Shiraz University Fetal Heart Sounds Database"],
                ["svdb", "MIT-BIH Supraventricular Arrhythmia Database"],
                ["szdb", "Post-Ictal Heart Rate Oscillations in Partial Epilepsy"],
                ["taichidb", "Tai Chi, Physiological Complexity, and Healthy Aging - Gait"],
                ["tappy", "Tappy Keystroke Data"],
                ["tns", "Surrogate Data with Correlations, Trends, and Nonstationarities"],
                ["tpehgdb", "Term-Preterm EHG Database"],
                ["tpehgt", "Term-Preterm EHG DataSet with Tocogram"],
                ["tremordb", "Effect of Deep Brain Stimulation on Parkinsonian Tremor"],
                ["twadb", "T-Wave Alternans Challenge Database"],
                ["ucddb", "St. Vincent's University Hospital / University College Dublin Sleep Apnea Database"],
                ["umwdb", "Long-term Recordings of Gait Dynamics"],
                ["unicaprop", "UniCA ElectroTastegram Database (PROP)"],
                ["vfdb", "MIT-BIH Malignant Ventricular Ectopy Database"],
                ["videopulse", "Video Pulse Signals in Stationary and Motion Conditions"],
                ["voiced", "VOice ICar fEDerico II Database"],
                ["wctecgdb", "Wilson Central Terminal ECG Database"],
                ["wctecgdb", "Wilson Central Terminal ECG Database"],
                ["wrist", "Wrist PPG During Exercise"],
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
        and save into `self._all_records` for further use

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
            self._all_records = wfdb.get_record_list(db_name or self.db_name)
        except:
            self._ls_rec_local()
            

    def _ls_rec_local(self,) -> NoReturn:
        """ finished, checked,

        find all records in `self.db_dir`
        """
        record_list_fp = os.path.join(self.db_dir, "RECORDS")
        if os.path.isfile(record_list_fp):
            with open(record_list_fp, "r") as f:
                self._all_records = f.read().splitlines()
                return
        print("Please wait patiently to let the reader find all records of the database from local storage...")
        start = time.time()
        self._all_records = get_record_list_recursive(self.db_dir, self.data_ext)
        print(f"Done in {time.time() - start:.3f} seconds!")
        with open(record_list_fp, "w") as f:
            for rec in self._all_records:
                f.write(f"{rec}\n")


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
        -----------
        items: str, or list of str, optional,
            the items to print,
            if not specified, then a comprehensive printing of meanings of all symbols will be performed

        References:
        -----------
        [1] https://archive.physionet.org/physiobank/annotations.shtml
        """
        attrs = vars(self)
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith("__") and func.endswith("__"))]

        beat_annotations = deepcopy(WFDB_Beat_Annotations)
        non_beat_annotations = deepcopy(WFDB_Non_Beat_Annotations)
        rhythm_annotations = deepcopy(WFDB_Rhythm_Annotations)

        all_annotations = [
            beat_annotations,
            non_beat_annotations,
            rhythm_annotations,
        ]

        summary_items = [
            "beat",
            "non-beat",
            "rhythm",
        ]

        if items is None:
            _items = ["attributes", "methods", "beat", "non-beat", "rhythm",]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if "attributes" in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if "methods" in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)
        if "beat" in _items:
            print("--- helpler - beat ---")
            pp.pprint(beat_annotations)
        if "non-beat" in _items:
            print("--- helpler - non-beat ---")
            pp.pprint(non_beat_annotations)
        if "rhythm" in _items:
            print("--- helpler - rhythm ---")
            pp.pprint(rhythm_annotations)

        for k in _items:
            if k in summary_items:
                continue
            for a in all_annotations:
                if k in a.keys() or "("+k in a.keys():
                    try:
                        print(f"{k.split('(')[1]} stands for {a[k]}")
                    except:
                        print(f"{k} stands for {a['('+k]}")


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
        self.fs = None
        self._all_records = None
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


    def safe_edf_file_operation(self, operation:str="close", full_file_path:Optional[str]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        operation: str, default "close",
            operation name, can be "open" and "close"
        full_file_path: str, optional,
            path of the file which contains the psg data,
            if not given, default path will be used
        """
        if operation == "open":
            if self.file_opened is not None:
                self.file_opened._close()
            self.file_opened = EdfReader(full_file_path)
        elif operation =="close":
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
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith("__") and func.endswith("__"))]

        if items is None:
            _items = ["attributes", "methods", ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if "attributes" in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if "methods" in _items:
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
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith("__") and func.endswith("__"))]

        if items is None:
            _items = ["attributes", "methods", ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if "attributes" in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if "methods" in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)


class AudioDataBase(_DataBase):
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
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith("__") and func.endswith("__"))]

        if items is None:
            _items = ["attributes", "methods", ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if "attributes" in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if "methods" in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)


class OtherDataBase(_DataBase):
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

        typical "db_dir":
        ------------------
            "E:\\notebook_dir\\ecg\\data\xxx\\"
            "/export/algo/wenh06/ecg_data/xxx/"
        """
        super().__init__(db_name=db_name, db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        # self._set_logger(prefix=None)

        self.fs = None
        self._all_records = None
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
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not (func.startswith("__") and func.endswith("__"))]

        if items is None:
            _items = ["attributes", "methods", ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if "attributes" in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if "methods" in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)



ECGWaveForm = namedtuple(
    typename="ECGWaveForm",
    field_names=["name", "onset", "offset", "peak", "duration"],
)
