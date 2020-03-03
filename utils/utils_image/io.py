"""
Module: utils_image
File: io.py
Author: wenhao71
Date: 2019/12/3
remarks: utilities for io of images
"""
import os
import numpy as np
from typing import Tuple, Union, Optional, Tuple
from random import shuffle

from utils.common import ArrayLike


__all__ = [
    "RandomImagePicker",
    "random_image_picker",
    "get_labeled_exif",
    "normalize_image",
]


class RandomImagePicker(object):
    """
    iterator
    """
    def __init__(self, directory:str):
        """
        """
        self.dir = directory
        if not os.path.isdir(self.dir):
            raise ValueError("Invalid directory")
        self.all_files = [
            os.path.join(self.dir, item) for item in os.listdir(self.dir)
        ]
        shuffle(self.all_files)
        self.counter = 0

    def __iter__(self):
        """
        """
        return self

    def __next__(self):
        """
        """
        if self.counter < len(self.all_files):
            self.counter += 1
            return self.all_files[self.counter]
        else:
            raise StopIteration()


def random_image_picker(directory:str):
    """
    generator
    """
    counter = 0
    all_files = [
        os.path.join(directory, item) for item in os.listdir(directory)
    ]
    shuffle(all_files)
    while counter < len(all_files):
        yield all_files[counter]
        counter += 1


#-----------------------------------------------------------------
# EXIF data
import PIL
from PIL.Image import Image
from PIL.ExifTags import TAGS, GPSTAGS


def get_labeled_exif(img: Union[Image, str]) -> dict:
    """

    Reference:
    ----------
    [1] https://www.exiv2.org/tags.html
    [2] http://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf
    [3] https://www.exif.org/
    [4] https://en.wikipedia.org/wiki/Exif
    """
    
    if isinstance(img, str):
        pil_img = PIL.Image.open(img)
    elif isinstance(img, Image):
        pil_img = img

    try:
        exif = pil_img._getexif() or {}
    except:
        exif = dict(pil_img.getexif()) or {}

    labeled = {}
    for (key, val) in exif.items():
        labeled[TAGS.get(key)] = val

    geotagging = {}
    if labeled.get("GPSInfo"):
        for (key, val) in GPSTAGS.items():
            if key in labeled["GPSInfo"]:
                geotagging[val] = labeled["GPSInfo"][key]
        labeled["GPSInfo"] = geotagging

    return labeled


def get_decimal_from_dms(dms:ArrayLike, ref:str) -> float:
    """
    """
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1] / 60.0
    seconds = dms[2][0] / dms[2][1] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)


def get_coordinates(geotags:dict) -> Tuple[float]:
    """
    """
    lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])

    lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

    return (lat,lon)


def exif_color_space(img: Image, verbose:int=0) -> str:
    """
    查看从文件读取的Image的color profile信息
    """
    exif = img._getexif() or {}
    if exif.get(0xA001) == 1 or exif.get(0x0001) == 'R98':
        img_cs = 'srgb'
        if verbose >= 1:
            print ('This image uses sRGB color space')
    elif exif.get(0xA001) == 2 or exif.get(0x0001) == 'R03':
        img_cs = 'adobe_rgb'
        if verbose >= 1:
            print ('This image uses Adobe RGB color space')
    elif exif.get(0xA001) is None and exif.get(0x0001) is None:
        img_cs = 'unspecified'
        if verbose >= 1:
            print ('Empty EXIF tags ColorSpace and InteropIndex')
    else:
        img_cs = 'unknown'
        if verbose >= 1:
            print ('This image uses UNKNOWN color space ({}, {})'.format(exif.get(0xA001),exif.get(0x0001)))
    return img_cs


#TODO: add utilities for writing exif data



# -------------------------------------------------

def normalize_image(img:np.ndarray, value_range:ArrayLike, resize_shape:Optional[Tuple[int,int]]=None, backend:str='skimage', **kwargs) -> np.ndarray:
    """
    Normalize an image by resizing it and rescaling its values

    Parameters:
    -----------
    img: ndarray,
        input image, in RGB
    value_range: array_like,
        range of values of output image, in a form similar to [min_value, max_value]
    resize_shape: tuple, optional,
        output image shape, of the form (w,h)
    
    Returns:
    --------
    normalized_img: ndarray
        resized and rescaled image
    """
    if backend == 'skimage':
        from skimage.transform import resize
    elif backend == 'cv2':
        from cv2 import resize
    dtype = kwargs.get("dtype", np.float32)
    verbose = kwargs.get("verbose", 0)

    img_max = np.max(img)
    img_min = np.min(img)
    normalized_img = (img - img_min) / (img_max - img_min)
    normalized_img = normalized_img * (value_range[1] - value_range[0]) + value_range[0]
    if resize_shape is not None:
        if backend == 'skimage':
            normalized_img = resize(normalized_img,
                        resize_shape,
                        order=3,
                        mode='constant',
                        preserve_range=True,
                        anti_aliasing=True)
        elif backend == 'cv2':
            normalized_img = resize(normalized_img, resize_shape[::-1])
    normalized_img = normalized_img.astype(dtype)
    return normalized_img
