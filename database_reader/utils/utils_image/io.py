"""
Module: utils_image
File: io.py
Author: wenhao71
Date: 2019/12/3
remarks: utilities for io of images
"""
import os
import numpy as np
from typing import Tuple, Union, Optional, Tuple, NoReturn
from random import shuffle
import cv2

from ..common import ArrayLike


__all__ = [
    "RandomImagePicker",
    "random_image_picker",
    "get_labeled_exif",
    "normalize_image",
    "synthesis_img",
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
    # elif isinstance(img, Image):
    else:
        pil_img = img

    try:
        exif = pil_img._getexif() or {}
    except:
        try:
            exif = dict(pil_img.getexif()) or {}
        except:
            raise ValueError("Invalid input image")

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


def synthesis_img(raw_img:np.ndarray, bkgd_img:np.ndarray, raw_mask:np.ndarray, save_path:Optional[str]=None, verbose:int=0):
    """ finished, checked,

    generate synthetic image using `raw_img` with background `bkgd_img`, where the unchanged foreground is given by `raw_mask`

    Parameters:
    -----------
    raw_img: ndarray,
        the source image
    bkgd_img: ndarray,
        the background image
    raw_mask: ndarray,
        the mask to distinguish the foreground of the source image
    save_path: str, optional,
        path to save the synthetic image
    verbose: int, default 0,

    Returns:
    --------
    sys_img: ndarray,
        the generated image
    """
    if verbose >= 2:
        if 'plt' not in dir():
            import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(raw_img[...,::-1])
        plt.show()
        plt.figure()
        plt.imshow(bkgd_img[...,::-1])
        plt.show()
        plt.figure()
        plt.imshow(raw_mask, cmap='gray')
        plt.show()
    
    refined_mask = np.where(raw_mask>125, np.ones_like(raw_mask,dtype=np.uint8), np.zeros_like(raw_mask,dtype=np.uint8))
    _, _ , _, refined_mask = _get_refined_mask(raw_img, refined_mask)
    
    if verbose >= 2:
        plt.figure()
        plt.imshow(refined_mask, cmap='gray')
        plt.show()
    
    x_range = np.sort(np.where(refined_mask.sum(axis=1)>0)[0])
    y_range = np.sort(np.where(refined_mask.sum(axis=0)>0)[0])
    x_len, y_len = x_range[-1] - x_range[0], y_range[-1] - y_range[0]
    ratio = 0.2
    x_min, x_max = max(0, int(x_range[0]-ratio*x_len)), min(raw_img.shape[1], int(x_range[-1]+ratio*x_len))
    y_min, y_max = max(0, int(y_range[0]-ratio*y_len)), min(raw_img.shape[0], int(y_range[-1]+ratio*y_len))
    cropped_img = raw_img[x_min:x_max,y_min:y_max]
    cropped_mask = refined_mask[x_min:x_max,y_min:y_max]
    
    if verbose >= 2:
        plt.figure()
        plt.imshow(cropped_img[...,::-1])
        plt.show()
        plt.figure()
        plt.imshow(cropped_mask, cmap='gray')
        plt.show()
        print(f"cropped_mask.shape = {cropped_mask.shape}")
        print(f"np.unique(cropped_mask) = {np.unique(cropped_mask)}")

    bkgd_ratio = max(1, cropped_img.shape[0]/bkgd_img.shape[0], cropped_img.shape[1]/bkgd_img.shape[1])
    cropped_bkgd = cv2.resize(bkgd_img, (int(bkgd_ratio*cropped_img.shape[1]), int(bkgd_ratio*cropped_img.shape[0])))
    cropped_bkgd_x_min = randint(0, cropped_bkgd.shape[0]-cropped_img.shape[0]+1)
    cropped_bkgd_y_min = randint(0, cropped_bkgd.shape[1]-cropped_img.shape[1]+1)
    cropped_bkgd = cropped_bkgd[cropped_bkgd_x_min:cropped_bkgd_x_min+cropped_img.shape[0], cropped_bkgd_y_min:cropped_bkgd_y_min+cropped_img.shape[1]]
    
    cropped_mask_3d = np.dstack((cropped_mask,cropped_mask,cropped_mask))
    if verbose >= 2:
        plt.figure()
        # plt.imshow((cropped_mask_3d*255).astype(np.uint8))
        plt.imshow(cropped_bkgd[...,::-1])
        plt.show()
        print(f"cropped_mask_3d.shape = {cropped_mask_3d.shape}")
    
    sys_img = np.where(cropped_mask_3d==1, cropped_img, cropped_bkgd)
    
    if save_path is not None:
        cv2.imwrite(save_path, sys_img)

    return sys_img


def _get_refined_mask(img:np.ndarray, raw_mask:np.ndarray) -> Tuple[np.ndarray]:
    """

    refine the `raw_mask` via CLAHE, Gaussian blur, thresholding, etc.

    Parameters:
    -----------
    img: ndarray,
        the source image
    raw_mask: ndarray,
        the raw mask to be refined

    Returns:
    --------
    img_grayscale_gblur, img_binary, img_close, refined_mask: ndarray,
        img_grayscale_gblur: the Guassian blured grayscale image
        img_binary: the binary image after global thresholding of `img_grayscale_gblur`
        img_close: the image after applying the closing morphological transformation
        refined_mask: the refined mask

    References:
    -----------
    [1] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html?highlight=clahe#clahe-contrast-limited-adaptive-histogram-equalization
    [2] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html?highlight=gaussianblur#gaussian-filtering
    [3] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    """
    cm_2_pxl = 50
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE
    tile_sz = max(1, int(2 * cm_2_pxl))
    tile_nb = max(1, int(gray.shape[1] / tile_sz))
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(tile_nb, tile_nb))
    img_grayscale = clahe.apply(gray)
    # plt.hist(img_grayscale.flatten(), bins=50)
    
    # Gaussian filter
    kernel_sz = max(1, int(0.2 * cm_2_pxl))
    kernel_sz = kernel_sz if kernel_sz % 2 == 1 else kernel_sz + 1
    img_grayscale_gblur = cv2.GaussianBlur(img_grayscale, (kernel_sz, kernel_sz), 0)
    
    # global thresholding
    threshold = 230
    _, img_binary = cv2.threshold(img_grayscale_gblur, threshold, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    elem_sz = max(1, int(0.25 * cm_2_pxl))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (elem_sz, elem_sz))
    img_close = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    refined_mask = cv2.bitwise_or(img_close, raw_mask)
    
    elem_sz = max(1, int(0.75 * cm_2_pxl))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (elem_sz, elem_sz))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return img_grayscale_gblur, img_binary, img_close, refined_mask
