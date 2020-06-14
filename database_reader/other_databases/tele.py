# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, NoReturn
from numbers import Real

from ..utils.common import ArrayLike
from ..base import OtherDataBase


__all__ = [
    "TELE",
]


class TELE(OtherDataBase):
    """

    TELE ECG Database from Harvard Dataverse

    ABOUT sleep_accel:
    ------------------
    1. Contains 250 telehealth ECG records (collected using dry metal electrodes) from 120 patients with annotated QRS and artifact masks
    2. The original dataset contains 300 records, 50 of which are discarded due to low quality
    3. The mains frequency was 50 Hz, the sampling frequency was 500 Hz
    4. The top and bottom rail voltages were 5.556912223578890 and -5.554198887532222 mV respectively
    5. Each record in the TELE database is stored as a X_Y.dat file where X indicates the index of the record in the TELE database and Y indicates the index of the record in the original dataset
    6. The .dat file is a comma separated values file. Each line contains:
        - the ECG sample value (mV)
        - a boolean indicating the locations of the annotated qrs complexes
        - a boolean indicating the visually determined mask
        - a boolean indicating the software determined mask

    NOTE:
    -----

    ISSUES:
    -------
    1. 

    Usage:
    ------
    1. ECG QRS delineation

    References:
    -----------
    [1] https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QTG0EP
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
        """
        super().__init__(db_name="TELE", db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
