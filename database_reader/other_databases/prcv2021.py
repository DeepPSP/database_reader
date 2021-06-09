# -*- coding: utf-8 -*-
import os, io
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd

from ..utils.common import (
    ArrayLike,
    get_record_list_recursive3,
)
from ..base import OtherDataBase


__all__ = [
    "PRCV2021",
]


class PRCV2021(OtherDataBase):
    """

    PRCV 2021 Alzheimer Disease Classification Competition

    ABOUT prcv2021
    --------------
    1. data are sMRI (structural Magnetic Resonance Imaging) data
    2. subjects are divided into three classes:
        - AD (Alzheimer Disease)
        - MCI (Mild Cognitive Impairment)
        - NC (Normal Control)
    3. columns in the stats csv file (TODO: find units for some columns (measurements)):
        - new_subject_id: subject id
        - site: 
        - age: age of the subject
        - male: boolean value indicating subject is male (1) or not (0)
        - female: boolean value indicating subject is female (1) or not (0)
        - NC: boolean value indicating subject is of class NC (1) or not (0)
        - MCI: boolean value indicating subject is of class MCI (1) or not (0)
        - AD: boolean value indicating subject is of class AD (1) or not (0)
        - Label: class (map) of the subject, 0 for NC, 1 for MCI, 2 for AD
        - Resolution: sMRI image resolution
        - Noise: noise (quality measures) of sMRI image evaluated using CAT12
        - Bias: bias (quality measures) of sMRI image evaluated using CAT12
        - IQR: weighted overall sMRI image quality evaluated using CAT12
        - TIV: total intracranial volume
        - CSF: cerebrospinal fluid
        - GMV: grey matter volume
        - WMV: white matter volume
        - Thickness: (mean of) cortical thickness
        - Thickness_std: standard deviation of cortical thickness

    NOTE
    ----

    ISSUES
    ------
    1. 

    Usage
    -----
    1. alzheimer disease classification

    References
    ----------
    [1] https://competition.huaweicloud.com/information/1000041489/introduction
    [2] http://www.neuro.uni-jena.de/cat/
    [3] http://www.neuro.uni-jena.de/cat12/CAT12-Manual.pdf
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs:Any) -> NoReturn:
        """ not finished,

        Parameters
        ----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments
        """
        super().__init__(db_name="PRCV2021", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self._all_records = get_record_list_recursive3(self.db_dir, "^Subject_[\d]{4}\.npy$")
        self.data_ext = "npy"
        self._train_dir = [os.path.dirname(item) for item in self._all_records]
        if len(self._train_dir) != 1:
            raise ValueError("records not in ONE directory")
        self._train_dir = self._train_dir[0]
        try:
            _stats_file = get_record_list_recursive3(self.db_dir, "^train_open\.csv$")[0]
        except:
            raise FileNotFoundError("stats file not found")
        self._stats = pd.read_csv(os.path.join(self.db_dir, f"{_stats_file}.csv"))


    @property
    def df_stats(self):
        """
        """
        return self._stats


    def get_subject_id(self, rec:str) -> int:
        """
        Attach a `subject_id` to the record, in order to facilitate further uses

        Parameters
        ----------
        rec: str,
            record name

        Returns
        -------
        sid: int,
            a `subject_id` attached to the record `rec`
        """
        sid = int(rec.replace("Subject_", ""))
        return sid



# from the officially provided npy_data_explain.csv
npy_data_explain_df = pd.read_csv(io.StringIO("""atlas_name,ROI_count,start_dim,end_dim
mesh.intensity_AAL.Template_T1_IXI555_MNI152_GS.csv,98,0,97
mesh.intensity_AAL2.Template_T1_IXI555_MNI152_GS.csv,102,98,199
mesh.intensity_AAL3v1.Template_T1_IXI555_MNI152_GS.csv,135,200,334
mesh.intensity_AICHA_reordered.Template_T1_IXI555_MNI152_GS.csv,373,335,707
mesh.intensity_Brodmann.Template_T1_IXI555_MNI152_GS.csv,41,708,748
mesh.intensity_Cerebellum-MNIflirt.Template_T1_IXI555_MNI152_GS.csv,6,749,754
mesh.intensity_Gordon.Template_T1_IXI555_MNI152_GS.csv,333,755,1087
mesh.intensity_Hammers_mith_83.Template_T1_IXI555_MNI152_GS.csv,75,1088,1162
mesh.intensity_Hammers_mith_95.Template_T1_IXI555_MNI152_GS.csv,87,1163,1249
mesh.intensity_Juelich_thr25.Template_T1_IXI555_MNI152_GS.csv,101,1250,1350
mesh.intensity_MIST_12.Template_T1_IXI555_MNI152_GS.csv,12,1351,1362
mesh.intensity_MIST_122.Template_T1_IXI555_MNI152_GS.csv,109,1363,1471
mesh.intensity_MIST_197.Template_T1_IXI555_MNI152_GS.csv,175,1472,1646
mesh.intensity_MIST_20.Template_T1_IXI555_MNI152_GS.csv,20,1647,1666
mesh.intensity_MIST_325.Template_T1_IXI555_MNI152_GS.csv,288,1667,1954
mesh.intensity_MIST_36.Template_T1_IXI555_MNI152_GS.csv,32,1955,1986
mesh.intensity_MIST_444.Template_T1_IXI555_MNI152_GS.csv,395,1987,2381
mesh.intensity_MIST_64.Template_T1_IXI555_MNI152_GS.csv,59,2382,2440
mesh.intensity_rBN_Atlas_246_1mm.Template_T1_IXI555_MNI152_GS.csv,236,2441,2676
mesh.intensity_Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,999,2677,3675
mesh.intensity_Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,998,3676,4673
mesh.intensity_Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,100,4674,4773
mesh.intensity_Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,100,4774,4873
mesh.intensity_Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,200,4874,5073
mesh.intensity_Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,200,5074,5273
mesh.intensity_HarvardOxford.Template_T1_IXI555_MNI152_GS.csv,109,5274,5382
mesh.intensity_MIST_7.Template_T1_IXI555_MNI152_GS.csv,7,5383,5389
mesh.intensity_Schaefer2018_300Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,300,5390,5689
mesh.intensity_Schaefer2018_700Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,700,5690,6389
mesh.intensity_Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,300,6390,6689
mesh.intensity_Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,400,6690,7089
mesh.intensity_Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,400,7090,7489
mesh.intensity_Schaefer2018_500Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,500,7490,7989
mesh.intensity_Schaefer2018_500Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,500,7990,8489
mesh.intensity_Schaefer2018_600Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,600,8490,9089
mesh.intensity_Schaefer2018_600Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,600,9090,9689
mesh.intensity_Schaefer2018_700Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,700,9690,10389
mesh.intensity_Schaefer2018_800Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,800,10390,11189
mesh.intensity_Schaefer2018_800Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,800,11190,11989
mesh.intensity_Schaefer2018_900Parcels_17Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,900,11990,12889
mesh.intensity_Schaefer2018_900Parcels_7Networks_order_FSLMNI152_1mm.Template_T1_IXI555_MNI152_GS.csv,899,12890,13788
mesh.intensity_Tian_Subcortex_S1_7T.Template_T1_IXI555_MNI152_GS.csv,12,13789,13800
mesh.intensity_Tian_Subcortex_S2_7T.Template_T1_IXI555_MNI152_GS.csv,22,13801,13822
mesh.intensity_Tian_Subcortex_S3_7T.Template_T1_IXI555_MNI152_GS.csv,36,13823,13858
mesh.intensity_Tian_Subcortex_S4_7T.Template_T1_IXI555_MNI152_GS.csv,42,13859,13900
mesh.intensity_Yeo2011_17Networks.Template_T1_IXI555_MNI152_GS.csv,17,13901,13917
mesh.intensity_Yeo2011_7Networks.Template_T1_IXI555_MNI152_GS.csv,7,13918,13924
MIST_36.csv,36,13925,13960
Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.csv,300,13961,14260
AAL.csv,116,14261,14376
AAL2.csv,120,14377,14496
AAL3v1.csv,166,14497,14662
AICHA_reordered.csv,384,14663,15046
brodmann.csv,41,15047,15087
Cerebellum-MNIflirt.csv,27,15088,15114
Gordon.csv,333,15115,15447
Hammers_mith_83.csv,83,15448,15530
Hammers_mith_95.csv,95,15531,15625
HarvardOxford.csv,113,15626,15738
Juelich_thr25.csv,103,15739,15841
MIST_12.csv,12,15842,15853
MIST_122.csv,122,15854,15975
MIST_197.csv,197,15976,16172
MIST_20.csv,20,16173,16192
MIST_325.csv,325,16193,16517
MIST_444.csv,444,16518,16961
MIST_64.csv,64,16962,17025
MIST_7.csv,7,17026,17032
rBN_Atlas_246_1mm.csv,246,17033,17278
Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_1mm.csv,1000,17279,18278
Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.csv,1000,18279,19278
Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm.csv,100,19279,19378
Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.csv,100,19379,19478
Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm.csv,200,19479,19678
Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.csv,200,19679,19878
Schaefer2018_300Parcels_17Networks_order_FSLMNI152_1mm.csv,300,19879,20178
Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.csv,400,20179,20578
Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.csv,400,20579,20978
Schaefer2018_500Parcels_17Networks_order_FSLMNI152_1mm.csv,500,20979,21478
Schaefer2018_500Parcels_7Networks_order_FSLMNI152_1mm.csv,500,21479,21978
Schaefer2018_600Parcels_17Networks_order_FSLMNI152_1mm.csv,600,21979,22578
Schaefer2018_600Parcels_7Networks_order_FSLMNI152_1mm.csv,600,22579,23178
Schaefer2018_700Parcels_17Networks_order_FSLMNI152_1mm.csv,700,23179,23878
Schaefer2018_700Parcels_7Networks_order_FSLMNI152_1mm.csv,700,23879,24578
Schaefer2018_800Parcels_17Networks_order_FSLMNI152_1mm.csv,800,24579,25378
Schaefer2018_800Parcels_7Networks_order_FSLMNI152_1mm.csv,800,25379,26178
Schaefer2018_900Parcels_17Networks_order_FSLMNI152_1mm.csv,900,26179,27078
Schaefer2018_900Parcels_7Networks_order_FSLMNI152_1mm.csv,900,27079,27978
Tian_Subcortex_S1_7T.csv,16,27979,27994
Tian_Subcortex_S2_7T.csv,34,27995,28028
Tian_Subcortex_S3_7T.csv,54,28029,28082
Tian_Subcortex_S4_7T.csv,62,28083,28144
Yeo2011_17Networks.csv,17,28145,28161
Yeo2011_7Networks.csv,7,28162,28168"""))
