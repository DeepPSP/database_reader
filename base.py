# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import pprint
import wfdb
import pandas as pd
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real
from utils.common import *


__all__ = [
    "PhysioNetDataBase",
    "NSRRDataBase",
    "ImageDataBases",
    "OtherDataBases",
]


class PhysioNetDataBase(object):
    """

    """
    def __init__(self, db_name:str, db_path:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_name: str,
            name of the database
        db_path: str, optional,
            storage path of the database,
            if not specified, `wfdb` will fetch data from the website of PhysioNet
        verbose: int, default 2,

        typical `db_path`:
        ------------------
            "E:\\notebook_dir\\ecg\\data\\PhysioNet\\xxx\\"
            "/export/algo/wenh06/ecg_data/xxx/"
        """
        self.db_name = db_name
        self.db_path = db_path
        self.freq = None
        self.all_records = None
        self.device_id = None  # maybe data are imported into impala db, to facilitate analyzing
        self.verbose = verbose

        if self.db_path is not None:
            if '/' in self.db_path:
                self.path_sep = '/'
            else:
                self.path_sep = '\\'
        else:
            self.path_sep = None
        
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
        

    def get_patient_id(self, rec:str) -> int:
        """
        Attach a `patient_id` to the record, in order to facilitate further uses

        Parameters:
        -----------
        rec: str,
            record name

        Returns:
        --------
        int, a `patient_id` attached to the record `rec`
        """
        return 0


    def database_info(self, detailed:bool=False) -> NoReturn:
        """
        print the information about the database

        detailed: bool, default False,
            if False, an short introduction of the database will be printed,
            if True, then docstring of the class will be printed additionally
        """
        return


    def helper(self, items:Union[List[str],str,type(None)]=None, **kwargs) -> NoReturn:
        """ finished, checked, to be improved,

        print corr. meanings of symbols belonging to `items`

        Parameters:
        items: str, or list of str, optional,
            the items to print,
            if not specified, then a comprehensive printing of meanings of all symbols will be performed
        """
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
            _items = ['beat', 'non-beat', 'rhythm']
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if 'beat' in _items:
            pp.pprint(beat_annotations)
        if 'non-beat' in _items:
            pp.pprint(non_beat_annotations)
        if 'rhythm' in _items:
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


class NSRRDataBase(object):
    """

    """
    def __init__(self, db_name:str, db_path:str, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_name: str,
            name of the database
        db_path: str,
            storage path of the database
        verbose: int, default 2,

        typical `db_path`:
        ------------------
            "E:\\notebook_dir\\ecg\\data\\NSRR\\xxx\\"
            "/export/algo/wenh06/ecg_data/NSRR/xxx/"
        """
        self.db_name = db_name
        self.db_path = db_path
        self.freq = None
        self.all_records = None
        self.device_id = None  # maybe data are imported into impala db, to facilitate analyzing
        self.verbose = verbose

        if '/' in self.db_path:
            self.path_sep = '/'
        else:
            self.path_sep = '\\'
        
        all_dbs = [
            ["shhs", "Multi-cohort study focused on sleep-disordered breathing and cardiovascular outcomes"],
            ["chat", "Multi-center randomized trial comparing early adenotonsillectomy to watchful waiting plus supportive care"],
            ["heartbeat", "Multi-center Phase II randomized controlled trial that evaluates the effects of supplemental nocturnal oxygen or Positive Airway Pressure (PAP) therapy"],
        ]
        self.df_all_db_info = pd.DataFrame(
            {
                "db_name": [item[0] for item in all_dbs],
                "db_description": [item[1] for item in all_dbs]
            }
        )
        self.kwargs = kwargs
        

    def get_patient_id(self, rec:str) -> int:
        """
        Attach a `patient_id` to the record, in order to facilitate further uses

        Parameters:
        -----------
        rec: str,
            record name

        Returns:
        --------
        int, a `patient_id` attached to the record `rec`
        """
        return 0


    def show_rec_stats(self, rec:str) -> NoReturn:
        """
        print the statistics about the record `rec`

        Parameters:
        -----------
        rec: str,
            record name
        """
        return


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


class ImageDataBases(object):
    """

    """
    def __init__(self, db_name:str, db_path:str, verbose:int=2, **kwargs):
        """
        """
        self.db_name = db_name
        self.db_path = db_path
        self.verbose = verbose


class OtherDataBase(object):
    """

    """
    def __init__(self, db_name:str, db_path:str, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_name: str,
            name of the database
        db_path: str,
            storage path of the database
        verbose: int, default 2,

        typical 'db_path':
        ------------------
            "E:\\notebook_dir\\ecg\\data\xxx\\"
            "/export/algo/wenh06/ecg_data/xxx/"
        """
        self.db_name = db_name
        self.db_path = db_path
        self.freq = None
        self.all_records = None
        self.device_id = None  # maybe data are imported into impala db, to facilitate analyzing
        self.verbose = verbose

        if '/' in self.db_path:
            self.path_sep = '/'
        else:
            self.path_sep = '\\'
        
        self.kwargs = kwargs
        

    def get_patient_id(self, rec:str) -> int:
        """
        Attach a `patient_id` to the record, in order to facilitate further uses

        Parameters:
        -----------
        rec: str,
            record name

        Returns:
        --------
        int, a `patient_id` attached to the record `rec`
        """
        return 0
    

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
