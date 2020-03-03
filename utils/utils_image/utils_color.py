# -*- coding: utf-8 -*-
"""
utilities for operations on color spaces for image processing

NOTE that there is an abuse of 'format' and 'color space'

TODO:
    implement the usage of other backends, epecially 'cv2', and 'pil' for acceleration;
    use color-math, colour-science(the most powerful one), skimage;
    use cython (or C/C++ lib) for acceleration?

    consistency of different backends!

experiment result: backend 'cv2' is 10-100 times faster than backend 'toy'
"""

import cv2
import colour
import numpy as np
from PIL import Image
from copy import deepcopy
from numbers import Real
from typing import Union, Optional, NoReturn

from utils import common
from ..common import ArrayLike, ArrayLike_Int, filter_by_percentile
from ..utils_universal import ConvexCone2D


__all__ = [
    "generate_pure_color_image",
    "compatible_imshow",
    "compatible_imread_cv2",
    "convert_color",
    "get_color_type",
]


_CVT_COLOR_BACKEND = 'cv2'
_AVAILABLE_CVT_COLOR_BACKENDS = [
    'cv2', 'pil', 'colour-science', 'toy',
]


def generate_pure_color_image(height:int, width:int, rgb_color:ArrayLike_Int, show:bool=True, **kwargs) -> np.ndarray:
    """ finished, checked,

    Create an RGB image of pure color `rgb_color`

    Parameters:
    -----------
    height: int,
        height of the image to be generated
    width: int,
        width of the image to be generated
    rgb_color: array_like of int,
        the standard format of RGB color, e.g. [100,100,100]
    show: bool, default True,
        whether to show the image generated or not
    kwargs: dict,
        additional arguments controlling the appearance of the printed image

    Returns:
    --------
    pure_color_image: ndarray,
        the pure color image in the RGB format
    """
    import matplotlib.pyplot as plt
    c = np.array(rgb_color)
    c_flatten = c.flatten()
    if len(c_flatten) != 3 or (c_flatten>255).any() or (c_flatten<0).any():
        raise ValueError("Invalid RGB color")
    pure_color_image = np.array([c for _ in range(height*width)],dtype=np.uint8).reshape((height,width,3))
    if show:
        figsize = kwargs.get('figsize', None)
        axis_off = kwargs.get('axis_off', True)
        plt.figure(figsize=figsize)
        plt.imshow(pure_color_image)
        if axis_off:
            plt.axis('off')
    return pure_color_image


def compatible_imshow(img_path:str, return_fmt:Optional[str]=None, **kwargs) -> Union[np.ndarray,NoReturn]:
    """ not finished, not checked,

    show the image correctly using cv2 and plt

    Parameters:
    -----------
    img_path: str,
        path of the image
    return_fmt: str, optional,
        return format (color space) of the image,
        implemented: RGB, BGR, CIEXYZ, CIELAB,
        if is None, the image will only be shown, but not returned
    kwargs: dict,
        additional arguments (`illuminant`, `observer`) for the following format:
        CIEXYZ, CIELAB;
        additional arguments controlling the appearance of the printed image

    Returns:
    --------
    rt_img: ndarray, or None
        the image in the format of `return_fmt`
    """
    import matplotlib.pyplot as plt
    bgr_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    figsize = kwargs.get('figsize', None)
    axis_off = kwargs.get('axis_off', True)
    plt.figure(figsize=figsize)
    plt.imshow(rgb_img)
    if axis_off:
        plt.axis('off')
    if return_fmt is None:
        return
    fmt = return_fmt.lower()
    if fmt == 'rgb':
        rt_img = rgb_img
    elif fmt == 'bgr':
        rt_img = bgr_img
    elif 'xyz' in fmt:
        rt_img = _rgb_to_ciexyz(rgb_img, **kwargs)
    elif 'lab' in fmt:
        rt_img = _rgb_to_cielab(rgb_img, **kwargs)
    elif 'gray' in fmt or 'grey' in fmt:
        rt_img = _rgb_to_grey(rgb_img, **kwargs)
    else:
        raise ValueError("format {} not implemented yet!".format(return_fmt))

    return rt_img


def compatible_imread_cv2(img_path:str, return_fmt:str) -> np.ndarray:
    """ not finished, not checked,

    read the image correctly in the format `return_fmt` using cv2

    Parameters:
    -----------
    img_path: str,
        path of the image
    return_fmt: str,
        return format (color space) of the image

    Returns:
    --------
    rt_img: ndarray, or None
        the image in the format (color space) of `return_fmt`
    """
    bgr_img = cv2.imread(img_path)
    cvt_operation = {
        "rgb": cv2.COLOR_BGR2RGB,
        "gray": cv2.COLOR_BGR2GRAY,
        "grey": cv2.COLOR_BGR2GRAY,
        'hsv': cv2.COLOR_BGR2HSV,
        'lab': cv2.COLOR_BGR2LAB,
        'xyz': cv2.COLOR_BGR2LUV,
    }
    rt_img = cv2.cvtColor(bgr_img,cvt_operation[return_fmt.lower()])

    return rt_img


#---------------------------------------------------------
# color converter


def convert_color(img:np.ndarray, src_fmt:str, dst_fmt:str, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ partly finished, partly checked,

    TODO: check compatibiliy of each backend

    Parameters:
    -----------
    img: ndarray,
        the image whose color space is to be converted from `src_fmt` to `dst_fmt`
    src_fmt: str,
        color space of `img`,
        can be one of GRAY(GREY), RGB, BGR, CIELAB, CIEXYZ, CIELUV, YCbCr, HSV, YIQ, CMYK,
    dst_fmt: str,
        the color space to be converted to,
        can be one of GRAY(GREY), RGB, BGR, CIELAB, CIEXYZ, CIELUV, YCbCr, HSV, YIQ, CMYK,
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be 'cv2', 'PIL', or 'JD',
        the backend to perform the color space conversion
    kwargs: dict,
        additional parameters
        `illuminant`, `observer` for the following format (color space):
        CIEXYZ, CIELAB, etc.
        `scale` for the following format (color space):
        HSV, CMYK, etc.

    Returns:
    --------
    dst_img: ndarray,
        the image in the format (color space) `dst_fmt`
    """
    src = src_fmt.lower()
    dst = dst_fmt.lower()
    err = ValueError("Color space conversion from {} to {} is not implemented yet".format(src_fmt, dst_fmt))
    if src == dst:
        dst_img = img.copy()
    elif src == 'rgb':
        if 'gray' in dst or 'grey' in dst:
            dst_img = _rgb_to_grey(img, backend=backend, **kwargs)
        if dst == 'bgr':
            dst_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif 'xyz' in dst:
            dst_img = _rgb_to_ciexyz(img, backend=backend, **kwargs)
        elif 'xyy' in dst:
            dst_img = _rgb_to_ciexyy(img, backend=backend, **kwargs)
        elif 'xy' in dst:  # 'xyz' not in dst
            dst_img = _rgb_to_ciexy(img, backend=backend, **kwargs)
        elif 'lab' in dst:
            dst_img = _rgb_to_cielab(img, backend=backend, **kwargs)
        elif 'luv' in dst:
            # dst_img = cv2.cvtColor(_rescale_rgb(img), cv2.COLOR_RGB2LUV)
            dst_img = _rgb_to_cieluv(img, backend=backend, **kwargs)
        elif 'ycbcr' in dst or 'ycrcb' in dst:
            # dst_img = cv2.cvtColor(_rescale_rgb(img), cv2.COLOR_RGB2YCrCb)
            dst_img = _rgb_to_ycbcr(img, backend=backend, **kwargs)
        elif 'hsv' in dst:
            dst_img = _rgb_to_hsv(img, backend=backend, **kwargs)
        elif 'yiq' in dst:
            dst_img = _rgb_to_yiq(img, backend=backend, **kwargs)
        elif 'cmyk' in dst:
            dst_img = _rgb_to_cmyk(img, backend=backend, **kwargs)
        else:
            raise err
    elif src == 'bgr':
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if dst == 'rgb':
            dst_img = rgb_img
        else:
            dst_img = convert_color(rgb_img, src_fmt='rgb', dst_fmt=dst_fmt, backend=backend, **kwargs)
    elif src in ['gray', 'grey']:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        dst_img = convert_color(rgb_img, src_fmt='rgb', dst_fmt=dst_fmt, backend=backend, **kwargs)
    elif 'xyz' in src:
        if 'rgb' in dst:
            dst_img = _ciexyz_to_rgb(img, backend=backend, **kwargs)
        elif 'lab' in dst:
            dst_img = _ciexyz_to_cielab(img, backend=backend, **kwargs)
        else:
            raise err
    elif 'xyy' in src:
        if dst == 'rgb':
            pass
    elif 'xy' in src:
        pass
    else:
        raise err

    return dst_img


def _rgb_to_grey(img:np.ndarray, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of gray levels

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, be converted to CIEXYZ
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion
    kwargs: dict,
        not used, only to be compatible with other color space conversion functions

    Returns:
    --------
    grey: ndarray,
        `img` in the format (color space) of grey levels

    References:
    -----------
    """
    if backend is None:
        backend = _CVT_COLOR_BACKEND
    if backend.lower() == 'cv2':
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif backend.lower() == 'colour-science':
        pass
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        pass
    else:
        raise ValueError("no backend named {} for converting color space from {} to {}".format(backend, 'RGB', 'GREY'))
    return grey


def _rgb_to_ciexyz(img:np.ndarray, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of CIEXYZ

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, be converted to CIEXYZ
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion
    kwargs: dict,
        not used, only to be compatible with other color space conversion functions

    Returns:
    --------
    cie_xyz: ndarray,
        `img` in the format (color space) of CIEXYZ

    References:
    -----------
    [1] https://www.w3.org/Graphics/Color/sRGB
    [2] https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    [3] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
    """
    if backend is None:
        backend = _CVT_COLOR_BACKEND
    if backend.lower() == 'cv2':
        cie_xyz = cv2.cvtColor(_rescale_rgb(img), cv2.COLOR_RGB2XYZ)
    elif backend.lower() == 'colour-science':
        colourspace = colour.utilities.first_item(colour.plotting.filter_RGB_colourspaces('sRGB').values())
        cie_xyz = colour.RGB_to_XYZ(img, colourspace.whitepoint, colourspace.whitepoint, colourspace.RGB_to_XYZ_matrix)
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        gamma_correction = lambda e: np.power((e+0.055)/1.055, 2.4) if e>0.04045 else e/12.92
        var_rgb = 100 * np.vectorize(gamma_correction)(_rescale_rgb(img))
        M = common.MAT_RGB_TO_CIEXYZ
        cie_xyz = np.apply_along_axis(lambda v:np.dot(M,v), -1, var_rgb)
        cie_xyz = cie_xyz.astype(np.float32)
    else:
        raise ValueError("no backend named {} for converting color space from {} to {}".format(backend, 'RGB', 'CIEXYZ'))
    return cie_xyz


def _rgb_to_ciexyy(img:np.ndarray, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of CIExyY

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, be converted to CIExyY
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion
    kwargs: dict,
        not used, only to be compatible with other color space conversion functions

    Returns:
    --------
    cie_xyy: ndarray,
        `img` in the format (color space) of CIExyY
    """
    if backend is None:
        # backend = _CVT_COLOR_BACKEND  # no such method in cv2
        backend = 'colour-science'
    if backend.lower() == 'cv2':
        pass
    elif backend.lower() == 'colour-science':
        cie_xyy = colour.XYZ_to_xyY(_rgb_to_ciexyz(img,backend='colour-science',**kwargs))
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        # default_white_point = np.array([0.3127, 0.3290, 0.0])
        # img_xyz = _rgb_to_ciexyz(img=img, backend=backend, **kwargs)
        pass
    return cie_xyy


def _rgb_to_ciexy(img:np.ndarray, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of CIExy

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, be converted to CIExy
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion
    kwargs: dict,
        not used, only to be compatible with other color space conversion functions

    Returns:
    --------
    cie_xy: ndarray,
        `img` in the format (color space) of CIExy
    """
    if backend is None:
        # backend = _CVT_COLOR_BACKEND  # no such method in cv2
        backend = 'colour-science'
    if backend.lower() == 'cv2':
        pass
    elif backend.lower() == 'colour-science':
        cie_xy = colour.XYZ_to_xy(_rgb_to_ciexyz(img,backend='colour-science',**kwargs))
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        # default_white_point = np.array([0.3127, 0.3290, 0.0])
        # img_xyz = _rgb_to_ciexyz(img=img, backend=backend, **kwargs)
        pass
    return cie_xy


def _ciexyz_to_cielab(img:np.ndarray, illuminant:str='D65', observer:int=2, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of CIEXYZ to the color space of CIELAB

    Parameters:
    -----------
    img: ndarray,
        the image whose color space is to be converted from CIEXYZ to CIELAB
    illuminant: str, default 'D65',
        ref. [2]
    observer: int, default 2,
        ref. [2]
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion

    Returns:
    --------
    cie_lab: ndarray,
        `img` in the format (color space) of CIELAB

    References:
    -----------
    [1] https://en.wikipedia.org/wiki/CIELAB_color_space#CIELAB%E2%80%93CIEXYZ_conversions
    [2] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    if backend is None:
        # backend = _CVT_COLOR_BACKEND  # no such method in cv2
        backend = 'toy'
    if backend.lower() == 'cv2':
        pass
    elif backend.lower() == 'colour-science':
        pass
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        delta = 6/29
        delta_cubic = delta**3
        a = 1/3/delta**2
        b = 4/29

        def _aux_func(t):
            return np.cbrt(t) if t > delta_cubic else a*t+b

        cie_lab = np.apply_along_axis(lambda v:np.array([116*_aux_func(v[1])-16, 500*(_aux_func(v[0])-_aux_func(v[1])), 200*(_aux_func(v[1])-_aux_func(v[2]))]), -1, img/common.ILLUMINANT_D65_2_XYZ)
        cie_lab = cie_lab.astype(np.float32)
    else:
        raise ValueError("no backend named {} for converting color space from {} to {}".format(backend, 'CIEXYZ', 'CIELAB'))
    return cie_lab


def _rgb_to_cielab(img:np.ndarray, illuminant:str='D65', observer:int=2, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of CIELAB

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, to be converted to CIELAB
    illuminant: str, default 'D65',
        ref. the function `_ciexyz_to_cielab`
    observer: int, default 2,
        ref. the function `_ciexyz_to_cielab`
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion

    Returns:
    --------
    cie_lab: ndarray,
        `img` in the format (color space) of CIELAB
    """
    if backend is None:
        backend = _CVT_COLOR_BACKEND
    if backend.lower() == 'cv2':
        cie_lab = cv2.cvtColor(_rescale_rgb(img), cv2.COLOR_RGB2LAB)
    elif backend.lower() == 'colour-science':
        pass
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        img = _validate_img_fmt(img, fmt='RGB')
        cie_xyz = _rgb_to_ciexyz(img)
        cie_lab = _ciexyz_to_cielab(cie_xyz, illuminant=illuminant, observer=observer, backend=backend)
    else:
        raise ValueError("no backend named {} for converting color space from {} to {}".format(backend, 'RGB', 'CIELAB'))
    return cie_lab


def _rgb_to_cieluv(img:np.ndarray, illuminant:str='D65', observer:int=2, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of CIELUV

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, to be converted to CIELUV
    illuminant: str, default 'D65',
        ref. the function `_ciexyz_to_cielab`
    observer: int, default 2,
        ref. the function `_ciexyz_to_cielab`
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be 'cv2', 'PIL', or 'JD',
        the backend to perform the color space conversion

    Returns:
    --------
    cie_luv: ndarray,
        `img` in the format (color space) of CIELUV
    """
    if backend is None:
        backend = _CVT_COLOR_BACKEND
    if backend.lower() == 'cv2':
        cieluv = cv2.cvtColor(_rescale_rgb(img), cv2.COLOR_RGB2LUV)
    elif backend.lower() == 'colour-science':
        pass
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        pass
    else:
        raise ValueError("no backend named {} for converting color space from {} to {}".format(backend, 'RGB', 'CIELUV'))
    return cieluv


def _rgb_to_ycbcr(img:np.ndarray, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of YCbCr

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, to be converted to YCbCr
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion

    Returns:
    --------
    cie_luv: ndarray,
        `img` in the format (color space) of CIELUV
    """
    if backend is None:
        backend = _CVT_COLOR_BACKEND
    if backend.lower() == 'cv2':
        ycbcr = cv2.cvtColor(_rescale_rgb(img), cv2.COLOR_RGB2YCrCb)
    elif backend.lower() == 'colour-science':
        pass
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        pass
    else:
        raise ValueError("no backend named {} for converting color space from {} to {}".format(backend, 'RGB', 'YCbCr'))
    return ycbcr


def _rgb_to_yiq(img:np.ndarray, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of YIQ

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, to be converted to YIQ
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion

    Returns:
    --------
    yiq: ndarray,
        `img` in the format (color space) of YIQ

    References:
    -----------
    [1] https://en.wikipedia.org/wiki/YIQ
    """
    if backend is None:
        # backend = _CVT_COLOR_BACKEND  # no such method in cv2
        backend = 'toy'
    if backend.lower() == 'cv2':
        pass
    elif backend.lower() == 'colour-science':
        pass
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        M = common.MAT_RGB_TO_YIQ
        yiq = np.apply_along_axis(lambda v:np.dot(M,v), -1, _rescale_rgb(img))
    else:
        raise ValueError("no backend named {} for converting color space from {} to {}".format(backend, 'RGB', 'YIQ'))
    return yiq


def _rgb_to_hsv(img:np.ndarray, scale:Real=1, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of HSV

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, to be converted to HSV
    scale: real number, default 360, can also be 1,
        the scale of H (hue), 0-360 or 0-1
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion

    Returns:
    --------
    hsv: ndarray,
        `img` in the format (color space) of HSV

    References:
    -----------
    [1] https://en.wikipedia.org/wiki/HSL_and_HSV
    """
    if backend is None:
        backend = _CVT_COLOR_BACKEND
    if scale not in [360, 1, 360.0, 1.0]:
        raise ValueError("Invalid scale")
    if backend.lower() == 'cv2':
        hsv = cv2.cvtColor(_rescale_rgb(img), cv2.COLOR_RGB2HSV)
    elif backend.lower() == 'colour-science':
        pass
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        pass
    if scale in [1.0, 1]:
        hsv[:,:,0] =  hsv[:,:,0]/360.0
    return hsv


def _rgb_to_cmyk(img:np.ndarray, scale:Real=1, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the color space of CMYK

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, to be converted to CMYK
    scale: real number, default 100, can also be 1.0,
        the scale of H (hue), 0-100 or 0.0-1.0
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion

    Returns:
    --------
    cmyk: ndarray,
        `img` in the format (color space) of CMYK

    References:
    -----------
    [1] https://en.wikipedia.org/wiki/
    """
    if backend is None:
        # backend = _CVT_COLOR_BACKEND  # no such method in cv2
        backend = 'toy'
    if scale not in [100, 1, 100.0, 1.0]:
        raise ValueError("Invalid scale")
    if backend.lower() == 'cv2':
        pass
    elif backend.lower() == 'colour-science':
        pass
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        nrows,ncols,_ = img.shape
        cmyk = np.zeros(shape=(nrows,ncols,4),dtype=np.float32)
        cmyk[:,:,:3] = 1.0 - _rescale_rgb(img)
        cmyk[:,:,3] = np.apply_along_axis(lambda v:np.min(v[:3]), -1, cmyk)

        cmyk[:,:,:3] = np.apply_along_axis(
            lambda v:np.array([0,0,0],dtype=np.float32) if v[3]==1.0 else (v[:3]-v[3])/(1-v[3]),
            axis=-1,
            arr=cmyk,
        )
    else:
        raise ValueError("no backend named {} for converting color space from {} to {}".format(backend, 'RGB', 'CMYK'))
    if scale in [100,100.0]:
        cmyk = 100.0*cmyk
    return cmyk


def _rgb_to_hex(img:np.ndarray, **kwargs) -> np.ndarray:
    """ not finished, finished part checked,

    convert `img` from the color space of RGB to the format of hex

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) RGB8, be converted to hex
    kwargs: dict,
        not used, only to be compatible with other color space conversion functions

    Returns:
    --------
    img_hex: ndarray,
        `img` in the format of hex
    """
    to_hex = lambda v: '#%02x%02x%02x' % tuple(v)
    img_hex = np.apply_along_axis(to_hex, axis=-1, arr=img,)
    return img_hex


def _ciexyz_to_rgb(img:np.ndarray, backend:Optional[str]=None, **kwargs) -> np.ndarray:
    """ not finished, not checked,

    convert `img` from the color space of CIEXYZ to the color space of RGB

    Parameters:
    -----------
    img: ndarray,
        the image, in the format (color space) CIEXYZ, be converted to RGB
    backend: str, default `_CVT_COLOR_BACKEND`, currently can be one in `_AVAILABLE_CVT_COLOR_BACKENDS`,
        the backend to perform the color space conversion
    kwargs: dict,
        not used, only to be compatible with other color space conversion functions

    Returns:
    --------
    rgb: ndarray,
        `img` in the format (color space) of RGB
    """
    if backend is None:
        # backend = _CVT_COLOR_BACKEND  # no such method in cv2
        backend = 'colour-science'
    if backend.lower() == 'cv2':
        pass
    elif backend.lower() == 'colour-science':
        rgb = colour.XYZ_to_sRGB(img)
        # srgb to 8bit RGB
    elif backend.lower() == 'pil':
        pass
    elif backend.lower() == 'toy':
        # default_white_point = np.array([0.3127, 0.3290, 0.0])
        # img_xyz = _rgb_to_ciexyz(img=img, backend=backend, **kwargs)
        pass
    return rgb


def _validate_img_fmt(img:np.ndarray, fmt:str, **kwargs) -> Union[np.ndarray, NoReturn]:
    """ not finished,

    check if `img` is a valid image in the format (color space) of `fmt`

    Parameters:
    -----------
    img: ndarray,
        the image to be validated
    fmt: str,
        format (color space) of `img`

    Returns:
    --------
    valid_img: ndarray,
        the validated image
    """
    if fmt.lower() == 'rgb':
        try:
            valid_img = _validate_rgb(img)
        except Exception as e:
            return
    return valid_img


def _validate_rgb(rgb_img:np.ndarray, verbose:int=0, **kwargs) -> Union[np.ndarray, NoReturn]:
    """ finished, checked,

    check if `rgb_img` is a valid RGB image

    Parameters:
    -----------
    rgb_img: ndarray,
        the image to be checked
    verbose: int, default 0, not used currently

    Returns:
    --------
    valid_img: ndarray,
        the validated RGB image
    """
    _,_,chn = rgb_img.shape
    if chn == 3 and ((rgb_img>=0) & (rgb_img<=255)).all():
        valid_img = rgb_img.astype(np.uint8)
    elif chn == 3 and ((rgb_img>=0) & (rgb_img<=1)).all():
        valid_img = rgb_img.astype(np.float32)
    else:
        raise ValueError("Invalid RGB image")
    return valid_img


def _rescale_rgb(rgb_img:np.ndarray, verbose:int=0) -> np.ndarray:
    """ finished, checked,

    if `rgb_img` is with values in 0.0-1.0, rescale it into values within 0-255,
    vice versa

    Parameters:
    -----------
    rgb_img: ndarray,
        the image to be rescaled
    verbose: int, default 0, not used currently

    Returns:
    --------
    rescaled_img: ndarray,
        the validated RGB image
    """
    rescaled_img = _validate_rgb(rgb_img)
    if rescaled_img.dtype == np.float32:
        rescaled_img = (rescaled_img * 255.0).astype(np.uint8)
    elif rescaled_img.dtype == np.uint8:
        rescaled_img = (rescaled_img / 255.0).astype(np.float32)
    return rescaled_img



#--------------------------------------------------------------
# some applications

def get_color_type(roi_pixels:ArrayLike, color_space:str, rule_func:callable, kw_rf:Optional[dict]=None, **kwargs) -> tuple:
    """

    Parameters:
    -----------
    roi_pixels: array_like,
        pixels of region of interest, in RGB
    color_space: str,
        the color space to make observation, can be 'CIEXYZ', 'CIExy', etc.
    rule_func: callable,
        the function to map the observation in `color_space` to names of color types
    kw_rf: dict, optional,
        key word arguments for `rule_func`
    kwargs: dict,
        other parameters, to write

    Returns:
    --------
    ret, dict, with items `centroid`, `color_type`
    """
    cs = color_space.lower()
    backend = kwargs.get('backend', None)
    verbose = kwargs.get('verbose', 0)
    if 'rgb' in cs.lower():  # no need to do color space conversion
        observation = deepcopy(roi_pixels)
    else:
        observation = convert_color(roi_pixels, src_fmt='rgb', dst_fmt=cs, backend=backend)
    _q = kwargs.get("q", 60)
    _observation = filter_by_percentile(observation, q=_q)
    if len(_observation) > 0:
        observation = _observation
    centroid = np.nanmean(observation, axis=0)
    
    if kw_rf is not None:
        color_type = rule_func(centroid, **kw_rf)
    else:
        color_type = rule_func(centroid)
    ret = {
        "centroid": centroid,
        "color_type": color_type,
    }

    if verbose >= 2:
        plot_func = kwargs.get('plot_func', None)
        plot_params = kwargs.get('plot_params', {})
        if plot_func is not None and callable(plot_func):
            plot_func(observation, **plot_params)

    return ret



#---------------------------------------------------------------------------
# class of color space

class ColorSpace(object):
    """
    """
    def __init__(self, name:str):
        """
        """
        self.name = name


#---------------------------------------------------------------------------
# functions and classes borrowed from the package `colour-science`
# to prevent from unexpected errors
# too complicated, not to do currently



#---------------------------------------------------
# illuminants, not finished

# https://docs.opencv.org/4.1.2/de/d25/imgproc_color_conversions.html

# https://en.wikipedia.org/wiki/Standard_illuminant
# https://en.wikipedia.org/wiki/Illuminant_D65
# illuminant: D60, D65, etc.
# observer: 2°, 10°
ILLUMINANT_D65_2_XYZ = np.array([95.0489, 100.0, 108.8840])
ILLUMINANT_D65_10_XYZ = np.array([94.8110, 100.0, 107.3040])
ILLUMINANT_D60_XYZ = np.array([96.4212, 100.0, 82.5188])


# https://en.wikipedia.org/wiki/RGB_color_model
# https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
MAT_RGB_TO_CIEXYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]],dtype=np.float32)
MAT_CIEXYZ_TO_RGB = np.linalg.inv(MAT_RGB_TO_CIEXYZ)
# https://en.wikipedia.org/wiki/YIQ
MAT_RGB_TO_YIQ = np.array([[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]])
MAT_YIQ_TO_RGB = np.linalg.inv(MAT_RGB_TO_YIQ)
# https://en.wikipedia.org/wiki/HSL_and_HSV


# color space ranges, not finished
RGB_RANGE = [[0,255],[0,255],[0,255]]
CIEXYZ_RANGE = []
CIELAB_RANGE = []
CIELUV_RANGE = []
HSV_RANGE = []
YIQ_RANGE = []
YCBCR_RANGE = []
CMYK_RANGE = []





def exif_color_space(img: Image.Image, verbose:int=0) -> str:
    """
    check the color profile (space) of an Image object read from file
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
    