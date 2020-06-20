# -*- coding: utf-8 -*-
"""
"""
import os
import numpy as np
import pandas as pd
import json
import cv2
from PIL import Image

from database_reader.base import ImageDataBase


__all__ = [
    "COCO"
]


class COCO2017(ImageDataBase):
    """
    """
    def __init__(self, db_path:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_path: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="COCO2017", db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)

