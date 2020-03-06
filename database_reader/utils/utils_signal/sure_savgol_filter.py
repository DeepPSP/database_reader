# -*- coding: utf-8 -*-
"""
Savitzky-Golay Filter with SURE(Stein's unbiased risk estimator)

currently very very slow!
"""

import numpy as np
from numbers import Number
from typing import Union, Optional, Any, Dict, List, Tuple

from ..common import ArrayLike
from .utils_signal import noise_std_estimator, uni_polyn_der, eval_uni_polyn


__all__ = [
    "fit_savgol_sure",
    "savgol_polyn_coeffs", "generate_savgol_matrix",
    "sure_savgol_objective_func", "reg_sure_savgol_objective_func",
]


def fit_savgol_sure(data:ArrayLike, orders:Union[int,List[int],Tuple[int]], radii:Union[int,List[int],Tuple[int]], mode:str='reflect', sure_reg:bool=False, verbose:int=0) -> np.ndarray:
    """ finished, to improve time complexity,

    best paramters for ecg denoising:
        orders=[1,5]
        radii=20ms(e.g. 5 for frequency=250)

    Paramters:
    ----------
    data: array_like,
        the signal to be filtered
    orders: int or sequence of int,
        orders of the filters
    radii: int or sequence of int,
        radii of the windows of the filters
    mode: str, default 'reflect',
    sure_reg: bool, default False,
    verbose: int, default 0,

    Returns:
    --------
    filtered: ndarray,
        the filtered signal

    Reference:
    ----------
        [1] Krishnan S R, Seelamantula C S. On the selection of optimum Savitzky-Golay filters[J]. IEEE transactions on signal processing, 2012, 61(2): 380-391.
    """
    filtered = []
    cost_func = reg_sure_savgol_objective_func if sure_reg else sure_savgol_objective_func

    if isinstance(orders, int) and isinstance(radii, int):
        # from scipy.signal import savgol_filter
        # return savgol_filter(data, window_length=2*radii+1, polyorder=orders)
        expanded_data = list(data[radii-1::-1]) + list(data) + list(data[-radii:])  # the reflect mode
        expanded_data = np.array(expanded_data)
        for idx, _ in enumerate(data):
            x = expanded_data[idx:idx+2*radii+1]
            filtered.append(savgol_polyn_coeffs(x, orders)[0][0])
        return np.array(filtered)
    
    if isinstance(orders, (list,tuple)) and isinstance(radii, (list,tuple)):
        raise ValueError('varying orders and radii at the same time is not allowed (implemented) currently')
    
    if isinstance(orders, int):
        # check validity of order and radius
        if orders < 0 or not all([isinstance(r,int) for r in radii]) or not (2*np.array(radii)+1>orders).all():
            raise ValueError('check if input orders and radii are valid')
        all_radii = list(range(min(radii), max(radii)+1))
        max_radius = max(all_radii)
        expanded_data = list(data[max_radius-1::-1]) + list(data) + list(data[-max_radius:])  # the reflect mode
        expanded_data = np.array(expanded_data)
        
        if verbose >= 1:
            print("all_radii are", all_radii)
            print("max_radius =", max_radius)
            print("len(data) = {0}, len(expanded_data) = {1}".format(len(data),len(expanded_data)))
            
        opt_radii = []
        for idx, _ in enumerate(data):
            x = expanded_data[idx:idx+2*max_radius+1]
            costs = [cost_func(orders, r, x, verbose) for r in all_radii]
            
            if verbose >= 2:
                print("costs =", costs)
            
            pos = np.argmin(costs)
            opt_radii.append(all_radii[pos])
            x = expanded_data[idx+max_radius-all_radii[pos]:idx+max_radius+all_radii[pos]+1]
            filtered.append(savgol_polyn_coeffs(x, orders)[0][0])
        if verbose >= 1:
            print("opt_radii =", opt_radii)
    else:  # radii is int
        if radii < 0 or not all([isinstance(o,int) for o in orders]) or not (2*radii+1>np.array(orders)).all():
            raise ValueError('check if input orders and radii are valid')
        all_orders = list(range(min(orders), max(orders)+1))
        all_orders = [item for item in all_orders if item%2==1]
        expanded_data = list(data[radii-1::-1]) + list(data) + list(data[-radii:])
        expanded_data = np.array(expanded_data)

        if verbose >= 1:
            print("all_orders are", all_orders)
            print("len(data) = {0}, len(expanded_data) = {1}".format(len(data),len(expanded_data)))

        opt_orders = []
        for idx, _ in enumerate(data):
            x = expanded_data[idx:idx+2*radii+1]
            costs = [cost_func(o, radii, x, verbose) for o in all_orders]
            
            if verbose >= 2:
                print("costs =", costs)
            
            pos = np.argmin(costs)
            opt_orders.append(all_orders[pos])
            filtered.append(savgol_polyn_coeffs(x,all_orders[pos])[0][0])
        if verbose >= 1:
            print("opt_orders =", opt_orders)

    filtered = np.array(filtered)
        
    return filtered


def generate_savgol_matrix(order:int, radius:int) -> np.ndarray:
    """

    Parameters:
    -----------
    order: int,
    radius: int,

    Returns:
    --------
    sm: ndarray,
        matrix of the filter of the given order and window radius
    """
    if 2*radius < order:
        raise ValueError('length of data must be larger than polynomial order')
    A = np.array([[np.power(m,p) for p in range(order+1)] for m in range(-radius,radius+1)])
    sm = np.linalg.inv(A.T@A)@A.T
    return sm


def savgol_polyn_coeffs(x:ArrayLike, order:int) -> Tuple[np.ndarray]:
    """ finished, checked,

    compute coefficients of the savitzky golay polynomial that best fit the data x

    Paramters:
    ----------
    x: array_like,
        the signal to be fitted to get the coefficients
    order: int,

    Returns:
    --------
    polyn_coeffs, H, tuple of ndarray,
        where polyn_coeffs, of shape (order+1,), are coefficients of the polynomial, in ascending order
        H is the matrix, such that polyn_coeffs = H x, of shape (order+1, len(x))
    """
    radius, remainder = divmod(len(x), 2)
    if remainder == 0:
        raise ValueError('length of data must be odd')
    if not isinstance(order,int) or order < 0:
        raise ValueError('order must be a non negative integer')
    if 2*radius < order:
        raise ValueError('length of data must be larger than polynomial order')
    
    A = np.array([[np.power(m,p) for p in range(order+1)] for m in range(-radius,radius+1)])

    # polyn_coeffs,_,_,_ = lstsq(A,x)
    H = np.linalg.inv(A.T@A)@A.T
    polyn_coeffs = H@np.array(x)

    return polyn_coeffs, H


def sure_savgol_objective_func(order:int, radius:int, data:ArrayLike, verbose:int=0) -> float:
    """ finished, checked,

    the SURE objective function with Savitzky-Golay filter for an instance centered in data

    Paramters:
    ----------
    order: int,
    radius: int,
    data: array_like,
    verbose: int, default 0,

    Returns:
    --------
    cost: float,

    Reference:
    ----------
        [1] Krishnan S R, Seelamantula C S. On the selection of optimum Savitzky-Golay filters[J]. IEEE transactions on signal processing, 2012, 61(2): 380-391.
    """
    halflen, remainder = divmod(len(data), 2)
    if remainder == 0:
        raise ValueError('data must be of odd length')
    if radius > halflen:
        raise ValueError('radius must not exceed half length of data')
    
    estimated_noise_var = np.power(noise_std_estimator(data),2)
    
    if verbose >= 1:
        print("estimated_noise_var =", estimated_noise_var)
    
    _x = data[halflen-radius:halflen+radius+1]
    f,H = savgol_polyn_coeffs(_x, order)
    # der_f = uni_polyn_der(f)
    f_x = eval_uni_polyn(list(range(-radius,radius+1)), f)
    # der_f_x = eval_uni_polyn(list(range(-radius,radius+1)), der_f)
    partial_f_x = np.array([eval_uni_polyn(i-radius, H[:,i]) for i in range(2*radius+1)])
    
    if verbose >= 1:
        print("_x =", _x)
        print("f =", f)
        # print("der_f =", der_f)
        print("f_x =", f_x)
        print("partial_f_x =", partial_f_x)
    
    cost = (np.sum(f_x*f_x - 2*f_x*_x) + 2*estimated_noise_var*np.sum(partial_f_x))/(2*radius+1)
    
    return cost


def reg_sure_savgol_objective_func(order:int, radius:int, data:ArrayLike, verbose:int=0) -> float:
    """ finished, checked
    
    the 'regularized' SURE objective function with Savitzky-Golay filter for an instance centered in data

    Paramters:
    ----------
    order: int,
    radius: int,
    data: array_like,
    verbose: int, default 0,

    Returns:
    --------
    cost: float,

    Reference:
    ----------
        [1] Krishnan S R, Seelamantula C S. On the selection of optimum Savitzky-Golay filters[J]. IEEE transactions on signal processing, 2012, 61(2): 380-391.
    """
    u = 1.2
    halflen, remainder = divmod(len(data), 2)
    if remainder == 0:
        raise ValueError('data must be of odd length')
    if radius > halflen:
        raise ValueError('radius must not exceed half length of data')
    
    estimated_noise_var = np.power(noise_std_estimator(data),2)
    
    if verbose >= 1:
        print("estimated_noise_var =", estimated_noise_var)
    
    _x = data[halflen-radius:halflen+radius+1]
    f,H = savgol_polyn_coeffs(_x,order)
    # der_f = uni_polyn_der(f)
    f_x = eval_uni_polyn(list(range(-radius,radius+1)), f)
    # der_f_x = eval_uni_polyn(list(range(-radius,radius+1)), der_f)
    partial_f_x = np.array([eval_uni_polyn(i-radius, H[:,i]) for i in range(2*radius+1)])
    
    if verbose >= 1:
        print("_x =", _x)
        print("f =", f)
        # print("der_f =", der_f)
        print("f_x =", f_x)
        print("partial_f_x =", partial_f_x)
    
    cost = (np.sum(f_x*f_x - 2*f_x*_x + u*estimated_noise_var*partial_f_x*partial_f_x) + 2*estimated_noise_var*np.sum(partial_f_x))/(2*radius+1)
    
    return cost
