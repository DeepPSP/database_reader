# -*- coding: utf-8 -*-
"""
"""
import os
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, List, NoReturn
from numbers import Real

from database_reader.utils.common import (
    ArrayLike,
    get_record_list_recursive,
)
from database_reader.base import ImageDataBase


__all__ = [
    "Hands11K"
]


class Hands11K(ImageDataBase):
    """

    11k Hands
    Gender recognition and biometric identification using a large dataset of hand images

    References:
    -----------
    [1] https://sites.google.com/view/11khands
    [2] https://github.com/mahmoudnafifi/11K-Hands
    """
    def __init__(self, db_dir:str, mask_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="11kHands", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.mask_dir = mask_dir


    def add_background(self, bg_dir:str, save_dir:Optional[str]=None) -> NoReturn:
        """
        add background to the images of 11k Hands, which can largely improve performance of DL models

        Parameters:
        -----------
        bg_dir: str,
            directory of the background images
        save_dir: str, optional,
            directory to save the augmented hand images,
            if not specified, `self.working_dir` will be used
        """
        save_dir = save_dir or os.path.join(self.working_dir, 'img_with_bg')
        raise NotImplementedError
