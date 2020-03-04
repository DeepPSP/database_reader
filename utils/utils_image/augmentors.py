# -*- coding: utf-8 -*-
"""
Module: utils_image
File: augmentors.py
Author: wenhao
remarks: utilities for image geometrical (pre-)processing, for the use of data augmentations, etc.

NOTE:
if train models using keras,
use ImageDataGenerator from keras.preprocessing.image instead via
>>> from keras.preprocessing.image import ImageDataGenerator

existing packages for image augmentation:
    igmaug
    Augmentor
    ...
"""

import cv2
import numpy as np
from numpy.random import randint, choice, uniform
from copy import deepcopy
from numbers import Real
from typing import Union, Optional, Any, Callable, Tuple, Dict, List, NoReturn

from utils import ArrayLike, ArrayLike_Int, ArrayLike_Float
from utils import angle_d2r


__all__ = [
    "image_augmentation",
    "add_background",
]


__DEFAULT_ROTATION_ANGLE = 16
__DEFAULT_SHIFT_RATIO = 0.08
__DEFAULT_SHEAR_ANGLE = 8
__DEFAULT_ZOOM_RATIO = 0.08


def image_augmentation(img:np.ndarray, num:int, aug_operations:Optional[List[str]]=None, verbose:int=0, **kwargs) -> List[np.ndarray]:
    """ finished, checked,

    perform augmentations on `img`

    Paramters:
    ----------
    img: ndarray,
        the image to be augmented
    num: int,
        times for each operation to be applied to `img`
    aug_operations: list of str, optional,
        augmentations to be applied on `img`,
        a sublist (the entire list if is None) of ['rotation', 'shift', 'shear', 'zoom',]
    verbose: int, default 0,

    Returns:
    --------
    img_augs: list of ndarray,
        the list consisting of augmented images of `img`

    For face2gene:
    --------------
    Each facial region is randomly augmented by rotation with a range of 5 degrees, small vertical and horizontal shifts (shift range of 0.05), shear transformation (shear range of 5π / 180) and random zoom (zoom range of 0.05) horizontal flip
    """
    img_augs = []

    legal_operations = ['rotation', 'shift', 'shear', 'zoom',]

    ops = aug_operations if aug_operations is not None else legal_operations

    _illegal_operations = [op for op in ops if op not in legal_operations]
    if len(_illegal_operations) == 1:
        raise ValueError("The operation `{}` on the input image is illegal!".format(_illegal_operations[0]))
    elif len(_illegal_operations) > 1:
        raise ValueError("The operation `{}` on the input image are illegal!".format(_illegal_operations))

    rotation_angle = kwargs.get('rotation_angle', __DEFAULT_ROTATION_ANGLE)
    shift_ratio = kwargs.get('shift_ratio', __DEFAULT_SHIFT_RATIO)
    shear_angle = kwargs.get('shear_angle', __DEFAULT_SHEAR_ANGLE)
    zoom_ratio = kwargs.get('zoom_ratio', __DEFAULT_ZOOM_RATIO)

    # rotation
    if 'rotation' in ops:
        img_augs += _image_rotation(img, angle=rotation_angle, num=num, verbose=verbose)
    
    # vertical and horizontal shifts
    if 'shift' in ops:
        img_augs += _image_shift(img, ratio=shift_ratio, num=num, verbose=verbose)
    
    # shear transformation
    if 'shear' in ops:
        img_augs += _shear_transformation(img, angle=shear_angle, num=num, verbose=verbose)
    
    # zoom
    if 'zoom' in ops:
        img_augs += _image_zoom(img, ratio=zoom_ratio, num=num, verbose=verbose)
    
    return img_augs



#--------------------------------------------------------------
# geometric transformation

def _image_rotation(img:np.ndarray, angle:Real, num:int, verbose:int=0) -> np.ndarray:
    """ finished, checked,

    rotate `img` `num` times with angles randomly chosen in the range of (-`abs(angle)`, `abs(angle)`)

    Parameters:
    -----------
    img: ndarray,
        the image to be rotated
    angle: real number,
        range (can be negative) of the angles to rotate
    num: int,
        times of rotations applied to `img`
    verbose: int, default 0,
    
    Returns:
    --------
    l_rotated, list of ndarray,
        the rotated images
    """
    l_rotated = []
    nrows,ncols = img.shape[:2]
    _angle = abs(angle)
    angles = uniform(-_angle, _angle, num)

    if verbose >= 1:
        print('angles = {}'.format(angles))
    
    for a in angles:
        M = cv2.getRotationMatrix2D((ncols/2,nrows/2),a,1)
        l_rotated.append(cv2.warpAffine(img,M,(ncols,nrows)))
    
    return l_rotated


def _image_shift(img:np.ndarray, ratio:float, num:int, verbose:int=0) -> np.ndarray:
    """ finished, checked,

    shift `img` `num` times with ratios randomly chosen in the range (-`abs(ratio)`, `abs(ratio)`)

    Parameters:
    -----------
    img: ndarray,
        the image to be shifted
    ratio: float,
        bound (can be negative) of the shift in x and y coordinates
    num: int,
        times of shifts applied to `img`
    verbose: int, default 0,
    
    Returns:
    --------
    l_shifted, list of ndarray,
        the shifted images
    """
    l_shifted = []
    nrows,ncols = img.shape[:2]
    _ratio = abs(ratio)
    shift_x,shift_y = int(_ratio*ncols), int(_ratio*nrows)
    c = _generate_coordinates(-shift_x,shift_x,-shift_y,shift_y)
    shifts = []
    for _ in range(num):
        shifts.append(next(c))
    
    if verbose >= 1:
        print('shift_y = {},shift_x = {}'.format(shift_y,shift_x))
        print('shifts = {}'.format(shifts))
    
    for x,y in shifts:
        M = np.array([[1,0,x],[0,1,y]], dtype=np.float32)
        l_shifted.append(cv2.warpAffine(img,M,(ncols,nrows)))
    
    return l_shifted


def _shear_transformation(img:np.ndarray, angle:Real, num:int, verbose:int=0) -> np.ndarray:
    """ finished, checked,

    apply horizontal shear transformation to `img` `num` times with angles randomly chosen in the range (-`abs(angle)`, `abs(angle)`)

    Parameters:
    -----------
    img: ndarray,
        the image to perform shear transformations
    angle: real number,
        range (can be negative) of the angles to shear
    num: int,
        times of shear transformations applied to `img`
    verbose: int, default 0,
    
    Returns:
    --------
    l_sheared, list of ndarray,
        the sheared images
    """
    l_sheared = []
    nrows,ncols = img.shape[:2]
    _angle = angle_d2r(angle)
    radians = uniform(-_angle, _angle, num)
    alphas = np.tan(radians)

    if verbose >= 1:
        print('radians = {}\nalphas = {}'.format(radians,alphas))
    
    for a in alphas:
        M = np.array([[1,a,0],[0,1,0]], dtype=np.float32)
        l_sheared.append(cv2.warpAffine(img,M,(ncols,nrows)))

    return l_sheared


def _image_zoom(img:np.ndarray, ratio:float, num:int, verbose:int=0) -> np.ndarray:
    """ finished, checked,

    zoom `img` `num` times with ratios randomly chosen in the range (-`abs(ratio)`, `abs(ratio)`)

    Parameters:
    -----------
    img: ndarray,
        the image to be zoomed
    ratio: float,
        bound (can be negative) of the zooming ratios, minused by 1
    num: int,
        times of zoomings applied to `img`
    verbose: int, default 0,
    
    Returns:
    --------
    l_shifted, list of ndarray,
        the shifted images
    """
    l_zoomed = []
    nrows,ncols = img.shape[:2]
    _ratio = abs(ratio)

    ratios = uniform(-_ratio, _ratio, num)

    if verbose >= 1:
        print('ratios = {}'.format(ratios))

    for r in ratios:
        if r == 0:
            l_zoomed.append(deepcopy(img))
            continue
        elif r > 0:
            half_len_y = int(ncols*r/(1+r)/2)
            half_len_x = int(nrows*r/(1+r)/2)
            pts1 = np.array([[half_len_x,half_len_y], [nrows-half_len_x,half_len_y], [half_len_x,ncols-half_len_y], [nrows-half_len_x,ncols-half_len_y]], dtype=np.float32)
            pts2 = np.array([[0,0], [nrows,0], [0,ncols], [nrows,ncols]], dtype=np.float32)
        elif r < 0:
            half_len_y = int(-ncols*r/2)
            half_len_x = int(-nrows*r/2)
            pts2 = np.array([[half_len_x,half_len_y], [nrows-half_len_x,half_len_y], [half_len_x,ncols-half_len_y], [nrows-half_len_x,ncols-half_len_y]], dtype=np.float32)
            pts1 = np.array([[0,0], [nrows,0], [0,ncols], [nrows,ncols]], dtype=np.float32)
        
        if verbose >= 2:
            print('for ratio = {},\npts1 = {}\npts2 = {}'.format(r, pts1, pts2))
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        l_zoomed.append(cv2.warpPerspective(img,M,(ncols,nrows)))

    return l_zoomed


def _generate_coordinates(x_low:int, x_high:int, y_low:int, y_high:int):
    """
    a generator that generates coordinates within the given range,
    used in `_image_shift`
    """
    seen = set()
    x, y = randint(x_low, x_high), randint(y_low, y_high)
    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(x_low, x_high), randint(y_low, y_high)
        while (x, y) in seen:
            x, y = randint(x_low, x_high), randint(y_low, y_high)


def add_border(img:np.ndarray, color:ArrayLike=[0,0,0], add_size:Union[int,ArrayLike_Int]=100) -> np.ndarray:
    """
    """
    if isinstance(add_size, int):
        add_x_l = add_x_r = add_y_u = add_y_d = add_size
    elif len(add_size) == 2:
        add_x_l = add_x_r = add_size[0]
        add_y_d = add_size = add_size[1]
    elif len(add_size) == 4:
        add_x_l, add_x_r, add_y_u, add_y_d = add_size
    
    # img_new = np.zeros(shape=(img.shape[0]+add_x_l+add_x_r,img.shape[1]+add_y_u+add_y_d,3), dtype=np.uint8)
    img_new = np.array([np.array(color) for _ in range((img.shape[0]+add_x_l+add_x_r)*(img.shape[1]+add_y_u+add_y_d))], dtype=np.uint8).reshape((img.shape[0]+add_x_l+add_x_r,img.shape[1]+add_y_u+add_y_d,3))

    for idx1 in range(img.shape[0]):
        for idx2 in range(img.shape[1]):
            img_new[add_x_l+idx1,add_y_u+idx2] = img[idx1,idx2]
    
    return img_new



#---------------------------------------------------
# other operations

def add_background(raw_img:np.ndarray, bkgd_img:np.ndarray, mask_func:callable, kw_mask_func:Optional[dict]=None, save_dst:Optional[str]=None, verbose:Union[int,str]=0, **kwargs) -> Union[np.ndarray, NoReturn]:
    """

    raw_img, bkgd_img in BGR format
    """
    crop_ratio = kwargs.get("crop_ratio", 0.2)

    if verbose >= 2:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(raw_img[...,::-1])
        plt.show()
        plt.figure()
        plt.imshow(bkgd_img[...,::-1])
        plt.show()
    
    if kw_mask_func is None:
        refined_mask = mask_func(raw_img)
    else:
        refined_mask = mask_func(raw_img, **kw_mask_func)
    
    if verbose >= 2:
        plt.figure()
        plt.imshow(refined_mask, cmap='gray')
        plt.show()
    
    x_range = np.sort(np.where(refined_mask.sum(axis=1)>0)[0])
    y_range = np.sort(np.where(refined_mask.sum(axis=0)>0)[0])
    x_len, y_len = x_range[-1] - x_range[0], y_range[-1] - y_range[0]

    x_min, x_max = max(0, int(x_range[0]-crop_ratio*x_len)), min(raw_img.shape[1], int(x_range[-1]+crop_ratio*x_len))
    y_min, y_max = max(0, int(y_range[0]-crop_ratio*y_len)), min(raw_img.shape[0], int(y_range[-1]+crop_ratio*y_len))
    cropped_img = raw_img[x_min:x_max,y_min:y_max]
    cropped_mask = refined_mask[x_min:x_max,y_min:y_max]
    
    if verbose >= 2:
        plt.figure()
        plt.imshow(cropped_img[...,::-1])
        plt.show()
        plt.figure()
        plt.imshow(cropped_mask, cmap='gray')
        plt.show()
        print("cropped_mask.shape =", cropped_mask.shape)
        print("np.unique(cropped_mask) =", np.unique(cropped_mask))

    bkgd_ratio = max(1, cropped_img.shape[0]/bkgd_img.shape[0], cropped_img.shape[1]/bkgd_img.shape[1])
    cropped_bkgd = cv2.resize(bkgd_img, (int(bkgd_ratio*cropped_img.shape[1]), int(bkgd_ratio*cropped_img.shape[0])))
    cropped_bkgd_x_min = randint(0, cropped_bkgd.shape[0]-cropped_img.shape[0]+1)
    cropped_bkgd_y_min = randint(0, cropped_bkgd.shape[1]-cropped_img.shape[1]+1)
    cropped_bkgd = cropped_bkgd[cropped_bkgd_x_min:cropped_bkgd_x_min+cropped_img.shape[0], cropped_bkgd_y_min:cropped_bkgd_y_min+cropped_img.shape[1]]
    
    cropped_mask_3d = np.dstack((cropped_mask,cropped_mask,cropped_mask))
    if verbose >= 2:
        plt.figure()
        plt.imshow(cropped_bkgd[...,::-1])
        plt.show()
        print("cropped_mask_3d.shape =", cropped_mask_3d.shape)
    
    sys_img = np.where(cropped_mask_3d==1, cropped_img, cropped_bkgd)
    
    if save_dst is None:
        return sys_img
    else:
        cv2.imwrite(save_dst, sys_img)
