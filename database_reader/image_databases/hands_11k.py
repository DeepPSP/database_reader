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
from database_reader.utils.utils_image import synthesis_img
from database_reader.base import ImageDataBase


__all__ = [
    "Hands11K"
]


class Hands11K(ImageDataBase):
    """

    11k Hands
    Gender recognition and biometric identification using a large dataset of hand images

    About 11k Hands:
    ----------------
    1. contains 11,076 hand images (1600 x 1200 pixels) of 190 subjects, of varying ages between 18 - 75 years old
    2. the metadata csv file contains: (1) the subject ID, (2) gender, (3) age, (4) skin color, and (5) a set of information of the captured hand, i.e. right- or left-hand, hand side (dorsal or palmar), and logical indicators referring to whether the hand image contains accessories, nail polish, or irregularities
    3. download links: https://sites.google.com/view/11khands#h.p_HS0BIeMrtWbo

    References:
    -----------
    [1] https://sites.google.com/view/11khands
    [2] https://github.com/mahmoudnafifi/11K-Hands
    """
    def __init__(self, db_dir:str, mask_dir:str, hand_info_path:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str,
            directory of the images of the database
        mask_dir: str,
            directory of raw masks of the images
        hand_info_path: str,
            the csv file that stores the metadata of the images
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="11kHands", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.mask_dir = mask_dir
        self.df_hand_info = pd.read_csv(hand_info_path)


    def add_background(self, bkgd_dir:str, save_dir:Optional[str]=None) -> NoReturn:
        """
        add background to the images of 11k Hands, which can largely improve performance of DL models

        Parameters:
        -----------
        bkgd_dir: str,
            directory of the background images
        save_dir: str, optional,
            directory to save the augmented hand images,
            if not specified, `self.working_dir` will be used
        """
        save_dir = save_dir or os.path.join(self.working_dir, 'img_with_bg')

        all_bkgd = [os.path.join(bkgd_dir, item) for item in os.listdir(bkgd_dir)]
        all_bkgd = [item for item in all_bkgd if os.path.isfile(item)]
        l_img_fn = self.df_hand_info['filename'].values.tolist()

        bkgd_seq = np.random.choice(all_bkgd, size=len(l_img_fn), replace=True)

        for idx, fn in enumerate(l_img_fn):
            try:
                raw_img = cv2.imread(os.path.join(self.db_dir, fn))[::-1,::-1,:]
                raw_mask = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE)[::-1,::-1]
                bkgd_img = cv2.imread(bkgd_seq[idx])
                synthesis_img(raw_img, bkgd_img, raw_mask, save_path=os.path.join(save_dir, fn), verbose=verbose)
            except Exception:
                print(f"error occurred when processing the {idx}-th image with filename {fn}")
