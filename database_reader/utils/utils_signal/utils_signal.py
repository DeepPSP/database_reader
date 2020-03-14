# -*- coding: utf-8 -*-
"""
utilities for signal processing, which numpy, scipy, etc. lack
"""
import numpy as np
import pywt
import scipy
from copy import deepcopy
from math import atan2, factorial
from scipy import interpolate
from scipy.signal import butter, lfilter
from collections import namedtuple
from numbers import Number, Real
from typing import Union, List, NamedTuple, Optional, Tuple
try:
    from numba import jit
except:
    from ..utils_misc import trivial_jit as jit

from ..common import ArrayLike, ArrayLike_Int


np.set_printoptions(precision=5,suppress=True)


__all__ = [
    "detect_peaks",
    "phasor_transform",
    "uni_polyn_der",
    "eval_uni_polyn",
    "noise_std_estimator",
    "lstsq_with_smoothness_prior",
    "compute_snr",
    "compute_snr_improvement",
    "is_ecg_signal",
    "WaveletDenoiseResult",
    "wavelet_denoise",
    "wavelet_rec_iswt",
    "rr_interval_to_2d_timeseries",
    "resample_irregular_timeseries",
    "resample_discontinuous_irregular_timeseries",
    "butter_bandpass",
    "butter_bandpass_filter",
    "MovingAverage",
]


WaveletDenoiseResult = namedtuple(
    typename='WaveletDenoiseResult',
    field_names=['is_ecg', 'amplified_ratio', 'amplified_signal', 'raw_r_peaks', 'side_len', 'wavelet_name', 'wavelet_coeffs']
)


def detect_peaks(x:ArrayLike,
                 mph:Union[int,float,type(None)]=None, mpd:int=1,
                 threshold:int=0, left_threshold:int=0, right_threshold:int=0,
                 edge:str='rising', kpsh:bool=False, valley:bool=False,
                 show:bool=False, ax=None,
                 verbose:int=0) -> np.ndarray:

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        abbr. for maximum (minimum) peak height
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        abbr. for minimum peak distance
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their neighbors within the range of mpd.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True
    """
    data = deepcopy(x)
    data = np.atleast_1d(data).astype('float64')
    if data.size < 3:
        return np.array([], dtype=int)
    
    if valley:
        data = -data
        if mph is not None:
            mph = -mph

    # find indices of all peaks
    dx = data[1:] - data[:-1]  # equiv to np.diff()

    # handle NaN's
    indnan = np.where(np.isnan(data))[0]
    if indnan.size:
        data[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))

    if verbose >= 1:
        print('before filtering by mpd = {}, and threshold = {}, ind = {}'.format(mpd, threshold, ind.tolist()))
        print('additionally, left_threshold = {}, right_threshold = {}, length of data = {}'.format(left_threshold, right_threshold, len(data)))
    
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]

    if verbose >= 1:
        print('after handling nan values, ind = {}'.format(ind.tolist()))
    
    # peaks are only valid within [mpb, len(data)-mpb[
    ind = np.array([pos for pos in ind if mpd<=pos<len(data)-mpd])
    
    if verbose >= 1:
        print('after fitering out elements too close to border by mpd = {}, ind = {}'.format(mpd, ind.tolist()))

    # first and last values of data cannot be peaks
    # if ind.size and ind[0] == 0:
    #     ind = ind[1:]
    # if ind.size and ind[-1] == data.size-1:
    #     ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[data[ind] >= mph]
    
    if verbose >= 1:
        print('after filtering by mph = {}, ind = {}'.format(mph, ind.tolist()))
    
    # remove peaks - neighbors < threshold
    _left_threshold = left_threshold if left_threshold > 0 else threshold
    _right_threshold = right_threshold if right_threshold > 0 else threshold
    if ind.size and (_left_threshold > 0 and _right_threshold > 0):
        # dx = np.min(np.vstack([data[ind]-data[ind-1], data[ind]-data[ind+1]]), axis=0)
        dx = np.max(np.vstack([data[ind]-data[ind+idx] for idx in range(-mpd, 0)]), axis=0)
        ind = np.delete(ind, np.where(dx < _left_threshold)[0])
        if verbose >= 2:
            print('from left, dx =', dx.tolist())
        dx = np.max(np.vstack([data[ind]-data[ind+idx] for idx in range(1, mpd+1)]), axis=0)
        ind = np.delete(ind, np.where(dx < _right_threshold)[0])
        if verbose >= 2:
            print('from right, dx =', dx.tolist())
    if verbose >= 1:
        print('after filtering by threshold, ind =', ind.tolist())
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(data[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (data[ind[i]] > data[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    
    ind = np.array([item for item in ind if data[item]==np.max(data[item-mpd:item+mpd+1])])

    if verbose >= 1:
        print('after filtering by mpd, ind =', ind.tolist())

    if show:
        if indnan.size:
            data[indnan] = np.nan
        if valley:
            data = -data
            if mph is not None:
                mph = -mph
        _plot(data, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """
    Plot results of the detect_peaks function, see its help.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


def phasor_transform(s:ArrayLike, rv:Real) -> np.ndarray:
    """ finished, checked,

    phasor transform, applied to s, with sensitivity controlled by rv

    Reference:
    ----------
        [1] Maršánová L, Němcová A, Smíšek R, et al. Automatic Detection of P Wave in ECG During Ventricular Extrasystoles[C]//World Congress on Medical Physics and Biomedical Engineering 2018. Springer, Singapore, 2019: 381-385.
    """
    return np.vectorize(atan2)(s,rv)


def compute_snr(original:ArrayLike, noised:ArrayLike) -> float:
    """
    computation of signal to noise ratio of the noised signal

    Parameters:
    -----------

    Returns:
    --------

    """
    snr = 10*np.log10(np.sum(np.power(np.array(original),2))/np.sum(np.power(np.array(original)-np.array(noised),2)))
    return snr


def compute_snr_improvement(original:ArrayLike, noised:ArrayLike, denoised:ArrayLike) -> float:
    """
    computation of the improvement of signal to noise ratio of the denoised signal,
    compared to the noised signal
    """
    return 10*np.log10(np.sum(np.power(np.array(original)-np.array(noised),2))/np.sum(np.power(np.array(original)-np.array(denoised),2)))


def uni_polyn_der(coeff:ArrayLike, order:int=1, coeff_asc:bool=True) -> np.ndarray:
    """ finished, checked,

    compute the order-th derivative of a univariate polynomial with real (int,float) coefficients,
    faster than np.polyder

    for testing speed:
    >>> from timeit import timeit
    >>> print(timeit(lambda : np.polyder([1,2,3,4,5,6,7],5), number=100000))
    >>> print(timeit(lambda : uni_polyn_der([1,2,3,4,5,6,7],5), number=100000))

    Parameters:
    -----------
    coeff: array like,
        coefficients of the univariate polynomial,
    order: non negative integer
        order of the derivative
    coeff_asc: bool
        coefficients in ascending order (a_0,a_1,...,a_n) or not (descending order, a_n,...,a_0)
    
    Returns:
    --------
    der: np.ndarray
        coefficients of the order-th derivative
    """
    dtype = float if any([isinstance(item, float) for item in coeff]) else int
    _coeff = np.array(coeff,dtype=dtype)
    polyn_deg = len(_coeff) - 1

    if order < 0 or not isinstance(order, int):
        raise ValueError('order must be a non negative integer')
    elif order == 0:
        return _coeff
    elif order > polyn_deg:
        return np.zeros(1).astype(dtype)
    
    if coeff_asc:
        tmp = np.array([factorial(n)/factorial(n-order) for n in range(order,polyn_deg+1)],dtype=int)
        der = _coeff[order:]*tmp
    else:
        der = uni_polyn_der(_coeff[::-1], order, coeff_asc=True)[::-1]
    return der


def eval_uni_polyn(x:Union[int,float,list,tuple,np.ndarray], coeff:ArrayLike, coeff_asc:bool=True) -> Union[int,float,np.ndarray]:
    """ finished, checked,

    Parameters:
    -----------

    Returns:
    --------

    evaluate x at the univariate polynomial defined by coeff
    """
    polyn_order = len(coeff)-1
    if len(coeff) == 0:
        raise ValueError('please specify a univariate polynomial!')
    
    if coeff_asc:
        if isinstance(x, (int,float)):
            return np.sum(np.array(coeff)*np.array([np.power(x,k) for k in range(polyn_order+1)]))
        else:
            return np.array([eval_uni_polyn(p, coeff) for p in x])
    else:
        return eval_uni_polyn(x, coeff[::-1], coeff_asc=True)


def noise_std_estimator(data:ArrayLike) -> float:
    """ finished, checked,

    median estimator for the unknown std of the noise

    Parameters:
    -----------

    Returns:
    --------

    Reference:
    ----------
        [1] Katkovnik V, Stankovic L. Instantaneous frequency estimation using the Wigner distribution with varying and data-driven window length[J]. IEEE Transactions on signal processing, 1998, 46(9): 2315-2325.
    """
    return np.median(np.abs(np.diff(data))) / 0.6745


def der_operator(responce_len:int, input_len:int, order:int) -> np.ndarray:
    """ not finished,

    Parameters:
    -----------

    Returns:
    --------

    derivation operator in matrix form
    """
    if responce_len+order > input_len:
        raise ValueError("responce_len+order should be no greater than input_len")

    raise NotImplementedError


def lstsq_with_smoothness_prior(data:ArrayLike) -> np.ndarray:
    """ not finished,

    Parameters:
    -----------

    Returns:
    --------

    Reference:
    ----------
        [1]. Sameni, Reza. "Online Filtering Using Piecewise Smoothness Priors: Application to Normal and Abnormal Electrocardiogram Denoising." Signal Processing 133.C (2017): 52-63. Web.
    """
    raise NotImplementedError


def generate_rr_interval(nb_beats:int, bpm_mean:Real, bpm_std:Real, lf_hf:float, lf_freq:float=0.1, hf_freq:float=0.25, lf_std:float=0.01, hf_std:float=0.01) -> np.ndarray:
    """ finished, not checked,

    Parameters:
    -----------

    Returns:
    --------
    
    """
    expected_rr_mean = 60 / bpm_mean
    expected_rr_std = 60 * bpm_std / (bpm_mean*bpm_mean)
    
    lf = lf_hf*np.random.normal(loc=lf_freq, scale=lf_std, size=nb_beats)  # lf power spectum
    hf = np.random.normal(loc=hf_freq, scale=hf_std, size=nb_beats)  # hf power spectum
    rr_power_spectrum = np.sqrt(lf + hf)
    
    # random (uniformly distributed in [0,2pi]) phases
    phases = np.vectorize(lambda theta: np.exp(2*1j*np.pi*theta))(np.random.uniform(low=0.0, high=2*np.pi, size=nb_beats))
    # real part of inverse FFT of complex spectrum
    raw_rr = np.real(np.fft.ifft(rr_power_spectrum*phases)) / nb_beats
    raw_rr_std = np.std(raw_rr)
    ratio = expected_rr_std/raw_rr_std
    rr = (raw_rr * ratio) + expected_rr_mean
    
    return rr


def is_ecg_signal(s:ArrayLike, freq:int, wavelet_name:str='db6', verbose:int=0) -> bool:
    """ finished, to be improved,

    Parameters:
    -----------
    s: array_like,
        the signal to be denoised
    freq: int,
        frequency of the signal `s`
    wavelet_name: str, default 'db6'
        name of the wavelet to use
    verbose: int, default 0,
        for detailedness of printing

    Returns:
    --------
    True if the signal `s` is valid ecg signal, else return False

    """
    sig_len = len(s)
    spacing = 1000/freq

    # constants for computation
    valid_rr = [200, 3000]  # ms, bpm 300 - 20
    reasonable_rr = [300, 1500]  # ms, bpm 40 - 200
    rr_samp_len = 5
    step_len = int(0.1*freq)  # 100ms
    window_radius = int(0.3*freq)  # 300ms
    slice_len = 2*window_radius  # for cutting out head and tails of the reconstructed signals

    high_confidence = 1.0
    low_confidence = 0.4

    is_ecg_confidence = 0
    is_ecg_confidence_threshold = 1.0
    
    if verbose >= 2:
        import matplotlib.pyplot as plt
        from ..common import DEFAULT_FIG_SIZE_PER_SEC
        # figsize=(int(DEFAULT_FIG_SIZE_PER_SEC*len(s)/freq), 6)

        print('(level 3 of) the wavelet in use looks like:')
        _, psi, x = pywt.Wavelet(wavelet_name).wavefun(level=3)
        _,ax = plt.subplots()
        ax.plot(x, psi)
        ax.set_title(wavelet_name+' level 3')
        plt.show()

    qrs_freqs = [10, 40]  # Hz
    qrs_levels = [int(np.ceil(np.log2(freq/qrs_freqs[-1]))), int(np.floor(np.log2(freq/qrs_freqs[0])))]
    if qrs_levels[0] > qrs_levels[-1]:
        qrs_levels = qrs_levels[::-1]

    tot_level = qrs_levels[-1]

    if pow(2,tot_level) > sig_len:
        # raise ValueError('length of signal is too short')
        print('length ({}) of signal is too short (should be at least {}) to perform wavelet denoising'.format(sig_len,pow(2,tot_level)))
        return False
    
    base_len = pow(2,tot_level)
    mult, res = divmod(sig_len, base_len)
    if res > 0:
        s_padded = np.concatenate((np.array(s), np.zeros((mult+1)*base_len-sig_len)))
    else:
        s_padded = np.array(s)

    if verbose >= 1:
        print('tot_level = {}, qrs_levels = {}'.format(tot_level, qrs_levels))
        print('sig_len = {}, padded length = {}'.format(sig_len, len(s_padded)-sig_len))
        print('shape of s_padded is', s_padded.shape)
    
    # perform swt
    coeffs = pywt.swt(
        data=s_padded,
        wavelet=wavelet_name,
        level=tot_level
    )

    # cAn = coeffs[0][0]
    coeffs = [ [np.zeros(s_padded.shape), e[1]] for e in coeffs ]
    # coeffs[0][0] = cAn

    zero_coeffs = [ [np.zeros(s_padded.shape), np.zeros(s_padded.shape)] for _ in range(tot_level) ]
    # zero_coeffs = [ [coeffs[i][0], np.zeros(s_padded.shape)] for i in range(tot_level) ]
    
    qrs_signals = []
    for lv in range(qrs_levels[0],qrs_levels[-1]+1):
        c_ = deepcopy(zero_coeffs)
        c_[tot_level-lv][1] = coeffs[tot_level-lv][1]
        # for cA_lv in range(1,lv):
        #     c_[tot_level-cA_lv][0] = c_[tot_level-lv][1]
        qrs_sig = pywt.iswt(coeffs=c_, wavelet=wavelet_name)[:sig_len]
        qrs_signals.append(qrs_sig)

        if verbose >= 2:
            default_fig_sz = 120
            line_len = freq * 25  # 25 seconds
            nb_lines = len(qrs_sig) // line_len
            for idx in range(nb_lines):
                c = qrs_sig[idx*line_len:(idx+1)*line_len]
                _, ax = plt.subplots(figsize=(default_fig_sz,6))
                ax.plot(c, label='level {}'.format(lv))
                ax.legend(loc='best')
                ax.set_title('level {}'.format(lv), fontsize=24)
                plt.show()
            c = qrs_sig[nb_lines*line_len:]  # tail left
            if len(c) > 0:
                fig_sz = int(default_fig_sz*(len(s)-nb_lines*line_len)/line_len)
                _, ax = plt.subplots(figsize=(fig_sz,6))
                ax.plot(c, label='level {}'.format(lv))
                ax.legend(loc='best')
                ax.set_title('level {}'.format(lv), fontsize=24)
                plt.show()

    qrs_power = np.power(np.sum(np.array(qrs_signals)[:,slice_len:-slice_len], axis=0), 2)
    qrs_amplitudes = []
    idx = window_radius
    while idx < len(qrs_power)-window_radius:
        qrs_seg = qrs_power[idx-window_radius:idx+window_radius+1]
        qrs_amplitudes.append(np.max(qrs_seg)-np.min(qrs_seg))
        idx += step_len
    qrs_amp = np.percentile(qrs_amplitudes, 50) * 0.5

    if verbose >= 1:
        print('qrs_amplitudes = {}\nqrs_amp = {}'.format(qrs_amplitudes, qrs_amp))

    raw_r_peaks = detect_peaks(
        x=qrs_power,
        mpd=step_len,
        threshold=qrs_amp,
        verbose=verbose
    )

    raw_rr_intervals = np.diff(raw_r_peaks)*spacing

    if verbose >= 1:
        print('raw_r_peaks = {}\nraw_rr_intervals = {}'.format(raw_r_peaks.tolist(), raw_rr_intervals.tolist()))
        s_ = s[slice_len:-slice_len]
        if verbose >= 2:
            default_fig_sz = 120
            line_len = freq * 25  # 25 seconds
            nb_lines = len(qrs_power) // line_len
            for idx in range(nb_lines):
                c = qrs_power[idx*line_len:(idx+1)*line_len]
                c_s_ = s_[idx*line_len:(idx+1)*line_len]
                _, ax = plt.subplots(figsize=(default_fig_sz,6))
                ax.plot(c, color='blue')
                c_r = [r for r in raw_r_peaks if idx*line_len<=r<(idx+1)*line_len]
                for r in c_r:
                    ax.axvline(r-idx*line_len, color='red', linestyle='dashed', linewidth=0.5)
                ax.set_title('QRS power', fontsize=24)
                ax2 = ax.twinx()
                ax2.plot(c_s_, color='green')
                plt.show()
            c = qrs_power[nb_lines*line_len:]  # tail left
            c_s_ = s_[nb_lines*line_len:]
            if len(c) > 0:
                fig_sz = int(default_fig_sz*(len(s)-nb_lines*line_len)/line_len)
                _, ax = plt.subplots(figsize=(fig_sz,6))
                ax.plot(c, color='blue')
                c_r = [r for r in raw_r_peaks if nb_lines*line_len<=r]
                for r in c_r:
                    ax.axvline(r-nb_lines*line_len, color='red', linestyle='dashed', linewidth=0.5)
                ax.set_title('QRS power', fontsize=24)
                ax2 = ax.twinx()
                ax2.plot(c_s_, color='green')
                plt.show()
            # _, ax = plt.subplots(figsize=figsize)
            # ax.plot(qrs_power, color='blue')
            # for r in raw_r_peaks:
            #     ax.axvline(r, color='red', linestyle='dashed', linewidth=0.5)
            # ax.set_title('QRS power', fontsize=20)
            # ax2 = ax.twinx()
            # ax2.plot(s[slice_len:-slice_len], color='green', linestyle='dashed')
            # plt.show()

    # TODO: compute entropy, std., etc. of raw_r_peaks
    # criteria 1: number of (r) peaks
    if spacing*len(qrs_power)/reasonable_rr[1] <= len(raw_r_peaks) <= spacing*len(qrs_power)/reasonable_rr[0]:
        is_ecg_confidence += high_confidence
    elif spacing*len(qrs_power)/valid_rr[1] <= len(raw_r_peaks) <= spacing*len(qrs_power)/valid_rr[0]:
        is_ecg_confidence += low_confidence
    # else: zero confidence

    # criteria 2: std of rr intervals
    raw_rr_std = np.std(raw_rr_intervals)
    # TODO: compute confidence level via std

    # criteria 3: sample entropy of rr intervals
    # raw_r_peaks_entropy = ent.sample_entropy(raw_rr_intervals, sample_length=rr_samp_len)[-1]
    # TODO: compute confidence level via sample entropy

    if verbose >= 1:
        print('overall is_ecg_confidence = {}'.format(is_ecg_confidence))
    
    return True if is_ecg_confidence >= is_ecg_confidence_threshold else False


def wavelet_denoise(s:ArrayLike, freq:int, wavelet_name:str='db6', amplify_mode:str='ecg', sides_mode:str='nearest', cval:int=0, verbose:int=0, **kwargs) -> NamedTuple:
    """ finished, to be improved,

    denoise and amplify (if necessary) signal `s`, using wavelet decomposition

    Parameters:
    -----------
    s: array_like,
        the signal to be denoised
    freq: int,
        frequency of the signal `s`
    wavelet_name: str, default 'db6'
        name of the wavelet to use
    amplify_mode: str, default 'ecg',
        amplification mode, can be one of 'ecg', 'qrs', 'all', 'none'
    sides_mode: str, default 'nearest',
        the way to treat the head and tail of the reconstructed (only if amplification is performed) signal,
        implemented modes: 'nearest', 'mirror', 'wrap', 'constant', 'no_slicing'
        not yet implemented mode(s): 'interp'
    cval: int, default 0,
        used only when `side_mode` is set 'constant'
    verbose: int, default 0,
        for detailedness of printing

    Returns:
    --------
        WaveletDenoiseResult, with field_names: 'is_ecg', 'amplified_ratio', 'amplified_signal', 'raw_r_peaks'
    
    TODO:

    """
    if amplify_mode not in ['ecg', 'qrs', 'all', 'none']:
        raise ValueError("Invalid amplify_mode! amplify_mode must be one of "
        "'ecg', 'qrs', 'all', 'none'.")
    if sides_mode not in ['nearest', 'mirror', 'wrap', 'constant', 'no_slicing', 'interp']:
        raise ValueError("Invalid sides_mode! sides_mode must be one of "
        "'nearest', 'mirror', 'wrap', 'constant', 'no_slicing', 'interp'.")

    sig_len = len(s)
    spacing = 1000/freq

    # constants for computation
    valid_rr = [200, 3000]  # ms, bpm 300 - 20
    reasonable_rr = [300, 1500]  # ms, bpm 40 - 200
    rr_samp_len = 5
    step_len = int(0.1*freq)  # 100ms
    qrs_radius = int(0.1*freq)  # 100ms
    window_radius = int(0.3*freq)  # 300ms
    slice_len = 2*window_radius  # for cutting out head and tails of the reconstructed signals

    # standard_ecg_amplitude = 1100  # muV
    # need_amplification_threshold = 500  # muV
    # now can be set de hors
    standard_ecg_amplitude = kwargs.get('standard_ecg_amplitude', 1100)
    need_amplification_threshold = kwargs.get('need_amplification_threshold', 500)

    high_confidence = 1.0
    low_confidence = 0.4

    is_ecg_confidence = 0
    is_ecg_confidence_threshold = 1.0
    
    if verbose >= 2:
        import matplotlib.pyplot as plt
        from ..common import DEFAULT_FIG_SIZE_PER_SEC
        # figsize=(int(DEFAULT_FIG_SIZE_PER_SEC*len(s)/freq), 6)

        print('(level 3 of) the wavelet used looks like:')
        _, psi, x = pywt.Wavelet(wavelet_name).wavefun(level=3)
        _,ax = plt.subplots()
        ax.plot(x, psi)
        ax.set_title(wavelet_name+' level 3')
        plt.show()

    qrs_freqs = [10, 40]  # Hz
    qrs_levels = [int(np.ceil(np.log2(freq/qrs_freqs[-1]))), int(np.floor(np.log2(freq/qrs_freqs[0])))]
    if qrs_levels[0] > qrs_levels[-1]:
        qrs_levels = qrs_levels[::-1]

    ecg_freqs = [0.5, 45]  # Hz
    ecg_levels = [int(np.floor(np.log2(freq/ecg_freqs[-1]))), int(np.ceil(np.log2(freq/ecg_freqs[0])))]
        
    # if qrs_only:
    #     tot_level = qrs_levels[-1]
    # else:
    #     tot_level = ecg_levels[-1]
    tot_level = ecg_levels[-1]+1

    if pow(2,tot_level) > sig_len:
        # raise ValueError('length of signal is too short')
        print('length ({}) of signal is too short (should be at least {}) to perform wavelet denoising'.format(sig_len,pow(2,tot_level)))
        ret = WaveletDenoiseResult(is_ecg=False, amplified_ratio=1.0, amplified_signal=deepcopy(s), raw_r_peaks=np.array([]), side_len=slice_len, wavelet_name=wavelet_name, wavelet_coeffs=[])
        return ret
    
    base_len = pow(2,tot_level)
    mult, res = divmod(sig_len, base_len)
    if res > 0:
        s_padded = np.concatenate((np.array(s), np.zeros((mult+1)*base_len-sig_len)))
    else:
        s_padded = np.array(s)

    if verbose >= 1:
        print('tot_level = {}, qrs_levels = {}, ecg_levels = {}'.format(tot_level, qrs_levels, ecg_levels))
        print('sig_len = {}, padded length = {}'.format(sig_len, len(s_padded)-sig_len))
        print('shape of s_padded is', s_padded.shape)
    
    # perform swt
    raw_coeffs = pywt.swt(
        data=s_padded,
        wavelet=wavelet_name,
        level=tot_level
    )

    # cAn = raw_coeffs[0][0]
    coeffs = [ [np.zeros(s_padded.shape), e[1]] for e in raw_coeffs ]
    # coeffs[0][0] = cAn

    zero_coeffs = [ [np.zeros(s_padded.shape), np.zeros(s_padded.shape)] for _ in range(tot_level) ]
    # zero_coeffs = [ [raw_coeffs[i][0], np.zeros(s_padded.shape)] for i in range(tot_level) ]
    
    qrs_signals = []
    for lv in range(qrs_levels[0],qrs_levels[-1]+1):
        c_ = deepcopy(zero_coeffs)
        c_[tot_level-lv][1] = coeffs[tot_level-lv][1]
        # for cA_lv in range(1,lv):
        #     c_[tot_level-cA_lv][0] = c_[tot_level-lv][1]
        qrs_sig = pywt.iswt(coeffs=c_, wavelet=wavelet_name)[:sig_len]
        qrs_signals.append(qrs_sig)

        if verbose >= 2:
            default_fig_sz = 120
            line_len = freq * 25  # 25 seconds
            nb_lines = len(qrs_sig) // line_len
            for idx in range(nb_lines):
                c = qrs_sig[idx*line_len:(idx+1)*line_len]
                _, ax = plt.subplots(figsize=(default_fig_sz,6))
                ax.plot(c, label='level {}'.format(lv))
                ax.legend(loc='best')
                ax.set_title('level {}'.format(lv), fontsize=24)
                plt.show()
            c = qrs_sig[nb_lines*line_len:]  # tail left
            if len(c) > 0:
                fig_sz = int(default_fig_sz*(len(s)-nb_lines*line_len)/line_len)
                _, ax = plt.subplots(figsize=(fig_sz,6))
                ax.plot(c, label='level {}'.format(lv))
                ax.legend(loc='best')
                ax.set_title('level {}'.format(lv), fontsize=24)
                plt.show()

    qrs_power = np.power(np.sum(np.array(qrs_signals)[:,slice_len:-slice_len], axis=0), 2)
    qrs_amplitudes = []
    idx = window_radius
    while idx < len(qrs_power)-window_radius:
        qrs_seg = qrs_power[idx-window_radius:idx+window_radius+1]
        qrs_amplitudes.append(np.max(qrs_seg)-np.min(qrs_seg))
        idx += step_len
    qrs_amp = np.percentile(qrs_amplitudes, 50) * 0.5

    if verbose >= 1:
        print('qrs_amplitudes = {}\nqrs_amp = {}'.format(qrs_amplitudes, qrs_amp))

    raw_r_peaks = detect_peaks(
        x=qrs_power,
        mpd=step_len,
        threshold=qrs_amp,
        verbose=verbose
    )

    raw_rr_intervals = np.diff(raw_r_peaks)*spacing

    if verbose >= 1:
        print('raw_r_peaks = {}\nraw_rr_intervals = {}'.format(raw_r_peaks.tolist(), raw_rr_intervals.tolist()))
        s_ = s[slice_len:-slice_len]
        if verbose >= 2:
            default_fig_sz = 120
            line_len = freq * 25  # 25 seconds
            nb_lines = len(qrs_power) // line_len
            for idx in range(nb_lines):
                c = qrs_power[idx*line_len:(idx+1)*line_len]
                c_s_ = s_[idx*line_len:(idx+1)*line_len]
                _, ax = plt.subplots(figsize=(default_fig_sz,6))
                ax.plot(c, color='blue')
                c_r = [r for r in raw_r_peaks if idx*line_len<=r<(idx+1)*line_len]
                for r in c_r:
                    ax.axvline(r-idx*line_len, color='red', linestyle='dashed', linewidth=0.5)
                ax.set_title('QRS power', fontsize=24)
                ax2 = ax.twinx()
                ax2.plot(c_s_, color='green')
                plt.show()
            c = qrs_power[nb_lines*line_len:]  # tail left
            c_s_ = s_[nb_lines*line_len:]
            if len(c) > 0:
                fig_sz = int(default_fig_sz*(len(s)-nb_lines*line_len)/line_len)
                _, ax = plt.subplots(figsize=(fig_sz,6))
                ax.plot(c, color='blue')
                c_r = [r for r in raw_r_peaks if nb_lines*line_len<=r]
                for r in c_r:
                    ax.axvline(r-nb_lines*line_len, color='red', linestyle='dashed', linewidth=0.5)
                ax.set_title('QRS power', fontsize=24)
                ax2 = ax.twinx()
                ax2.plot(c_s_, color='green')
                plt.show()
            # _, ax = plt.subplots(figsize=figsize)
            # ax.plot(qrs_power, color='blue')
            # for r in raw_r_peaks:
            #     ax.axvline(r, color='red', linestyle='dashed', linewidth=0.5)
            # ax.set_title('QRS power', fontsize=20)
            # ax2 = ax.twinx()
            # ax2.plot(s[slice_len:-slice_len], color='green', linestyle='dashed')
            # plt.show()

    # TODO: compute entropy, std., etc. of raw_r_peaks
    # criteria 1: number of (r) peaks
    if spacing*len(qrs_power)/reasonable_rr[1] <= len(raw_r_peaks) <= spacing*len(qrs_power)/reasonable_rr[0]:
        is_ecg_confidence += high_confidence
    elif spacing*len(qrs_power)/valid_rr[1] <= len(raw_r_peaks) <= spacing*len(qrs_power)/valid_rr[0]:
        is_ecg_confidence += low_confidence
    # else: zero confidence

    # criteria 2: std of rr intervals
    raw_rr_std = np.std(raw_rr_intervals)
    # TODO: compute confidence level via std

    # criteria 3: sample entropy of rr intervals
    # raw_r_peaks_entropy = ent.sample_entropy(raw_rr_intervals, sample_length=rr_samp_len)[-1]
    # TODO: compute confidence level via sample entropy

    if verbose >= 1:
        print('overall is_ecg_confidence = {}'.format(is_ecg_confidence))
    
    if is_ecg_confidence >= is_ecg_confidence_threshold:
        qrs_amplitudes = []
        # note that raw_r_peaks are computed from qrs_power,
        #  which is sliced at head (and at tail) by slice_len
        raw_r_peaks = raw_r_peaks + slice_len
        for r in raw_r_peaks:
            qrs_seg = s[r-qrs_radius:r+qrs_radius+1]
            qrs_amplitudes.append(np.max(qrs_seg)-np.min(qrs_seg))
        qrs_amp = np.percentile(qrs_amplitudes, 75)
        if qrs_amp < need_amplification_threshold:
            amplify_ratio = standard_ecg_amplitude / qrs_amp
        else:
            amplify_ratio = 1.0

        if amplify_mode != 'none' and amplify_ratio > 1.0:
            c_ = deepcopy(coeffs)  # or deepcopy(zero_coeffs)?
            # c_ = deepcopy(zero_coeffs)

            if amplify_mode == 'ecg':
                levels_in_use = [ecg_levels[0], ecg_levels[-1]-2]
            elif amplify_mode == 'qrs':
                levels_in_use = [qrs_levels[0]-1, qrs_levels[-1]+1]
            elif amplify_mode == 'all':
                levels_in_use = [1, ecg_levels[-1]+1]
            # for lv in range(qrs_levels[0]-1, qrs_levels[-1]+2):
            # for lv in range(qrs_levels[0]-1, qrs_levels[-1]+1):
            # for lv in range(ecg_levels[0], ecg_levels[-1]+1):
            for lv in range(levels_in_use[0], levels_in_use[1]):
                c_[tot_level-lv][1] = amplify_ratio*coeffs[tot_level-lv][1]
            
            s_rec = pywt.iswt(coeffs=c_, wavelet=wavelet_name)[:sig_len]
            # s_rec = np.vectorize(lambda n: int(round(n)))(s_rec[slice_len:-slice_len])
            s_rec = np.vectorize(lambda n: int(round(n)))(s_rec)

            # add head and tail
            if sides_mode == 'nearest':
                s_rec[:slice_len] = s_rec[slice_len]
                s_rec[-slice_len:] = s_rec[-slice_len-1]
            elif sides_mode == 'mirror':
                s_rec[:slice_len] = s_rec[2*slice_len-1:slice_len-1:-1]
                s_rec[-slice_len:] = s_rec[-slice_len-1:-2*slice_len-1:-1]
            elif sides_mode == 'wrap':
                s_rec[:slice_len] = s_rec[-2*slice_len:-slice_len] + (s_rec[slice_len]-s_rec[slice_len-1])
                s_rec[-slice_len:] = s_rec[slice_len:2*slice_len] + (s_rec[-slice_len]-s_rec[-slice_len-1])
            elif sides_mode == 'constant':
                s_rec[:slice_len] = cval
                s_rec[-slice_len:] = cval
            elif sides_mode == 'no_slicing':
                pass  # do nothing to head and tail of s_rec
            elif sides_mode == 'interp':
                raise ValueError("Invalid sides_mode! sides_mode 'interp' not implemented yet!")
        else: # set no amplification, or need no amplification
            levels_in_use = [np.nan, np.nan]
            s_rec = deepcopy(s)
        
        if verbose >= 1:
            print('levels used for the purpose of amplification are {} to {} (inclusive)'.format(levels_in_use[0], levels_in_use[1]-1))
            print('amplify_ratio = {}\nqrs_amplitudes = {}'.format(amplify_ratio, qrs_amplitudes))
            if verbose >= 2:
                default_fig_sz = 120
                line_len = freq * 25  # 25 seconds
                nb_lines = len(s_rec) // line_len
                for idx in range(nb_lines):
                    c_rec = s_rec[idx*line_len:(idx+1)*line_len]
                    c = s[idx*line_len:(idx+1)*line_len]
                    _, ax = plt.subplots(figsize=(default_fig_sz,6))
                    ax.plot(c_rec,color='red')
                    ax.plot(c,alpha=0.6)
                    ax.set_title('signal amplified', fontsize=24)
                    c_r = [r for r in raw_r_peaks if idx*line_len<=r<(idx+1)*line_len]
                    for r in c_r:
                        ax.axvline(r-idx*line_len, color='red', linestyle='dashed', linewidth=0.5)
                    plt.show()
                c_rec = s_rec[nb_lines*line_len:]  # tail left
                c = s[nb_lines*line_len:]
                if len(c) > 0:
                    fig_sz = int(default_fig_sz*(len(s)-nb_lines*line_len)/line_len)
                    _, ax = plt.subplots(figsize=(fig_sz,6))
                    ax.plot(c_rec,color='red')
                    ax.plot(c,alpha=0.6)
                    ax.set_title('signal amplified', fontsize=24)
                    c_r = [r for r in raw_r_peaks if nb_lines*line_len<=r]
                    for r in c_r:
                        ax.axvline(r-nb_lines*line_len, color='red', linestyle='dashed', linewidth=0.5)
                    plt.show()
        
        ret = WaveletDenoiseResult(is_ecg=True, amplified_ratio=amplify_ratio, amplified_signal=s_rec, raw_r_peaks=raw_r_peaks, side_len=slice_len, wavelet_name=wavelet_name, wavelet_coeffs=raw_coeffs)
    else:  # not ecg
        raw_r_peaks = raw_r_peaks + slice_len
        ret = WaveletDenoiseResult(is_ecg=False, amplified_ratio=np.nan, amplified_signal=deepcopy(s), raw_r_peaks=raw_r_peaks, side_len=slice_len, wavelet_name=wavelet_name, wavelet_coeffs=raw_coeffs)
    
    return ret


def wavelet_rec_iswt(coeffs:List[List[np.ndarray]], levels:ArrayLike_Int, wavelet_name:str, verbose:int=0) -> np.ndarray:
    """ finished, checked,

    reconstruct signal, using pywt.iswt, using coefficients obtained by pywt.swt of level in `levels`

    Parameters:
    -----------
    coeffs: list of list (pair) of np.ndarray,
        wavelet ceofficients (list of [cA_n,cD_n], ..., [cA_1,cD_1]), obtained by pywt.swt
    levels: list of int,
        the levels to reconstruct from
    wavelet_name: str,
        name of the wavelet
    verbose: int, default 0,
        the detailedness of printing

    Returns:
    --------
    np.ndarray, the reconstructed signal
    """
    if verbose >= 2:
        import matplotlib.pyplot as plt
    
    sig_shape = coeffs[0][0].shape
    nb_levels = len(coeffs)

    if verbose >= 1:
        print('sig_shape = {}, nb_levels = {}'.format(sig_shape, nb_levels))
    
    if (nb_levels < np.array(levels)).any():
        raise ValueError('Invalid levels')
    
    c_ = [[np.zeros(sig_shape),np.zeros(sig_shape)] for _ in range(nb_levels)]
    for lv in levels:
        c_[nb_levels-lv][1] = coeffs[nb_levels-lv][1]
    sig_rec = pywt.iswt(coeffs=c_, wavelet=wavelet_name)

    if verbose >= 2:
        _, ax = plt.subplots(figsize=(20,4))
        ax.plot(sig_rec)
        plt.show()
    
    return sig_rec


def rr_interval_to_2d_timeseries(rr_intervals:ArrayLike_Int) -> np.ndarray:
    """ finished, checked,

    transform the 1d array of rr intervals to a 2d irregular timeseries

    Parameters:
    rr_intervals: array_like,
        the rr intervals, with units in ms
    
    Returns:
    --------
    2d array, each element in the form of [time, value]
    """
    ts = np.append(0,np.cumsum(np.array(rr_intervals))[:-1])
    return np.column_stack((ts,rr_intervals))


def resample_irregular_timeseries(s:ArrayLike, output_fs:Real=2, method:str="spline", return_with_time:bool=False, tnew:Optional[ArrayLike]=None, options:dict={}, verbose:int=0) -> np.ndarray:
    """ finished, checked,

    resample the 2d irregular timeseries `s` into a 1d or 2d regular time series with frequency `output_fs`,
    elements of `s` are in the form [time, value], where the unit of `time` is ms

    Parameters:
    -----------
    s: array_like,
        the 2d irregular timeseries
    output_fs: Real, default 2,
        the frequency of the output 1d regular timeseries
    method: str, default "spline"
        interpolation method, can be 'spline' or 'interp1d'
    return_with_time: bool, default False,
        return a 2d array, with the 0-th coordinate being time
    tnew: array_like, optional,
        the array of time of the output array
    options: dict, default {},
        additional options for the corresponding methods in scipy.interpolate

    Returns:
    --------
    np.ndarray, a 1d or 2d regular time series with frequency `output_freq`

    NOTE:
    pandas also has the function to regularly resample irregular timeseries
    """
    if method not in ["spline", "interp1d"]:
        raise ValueError("method {} not implemented".format(method))

    if verbose >= 1:
        print("len(s) = {}".format(len(s)))

    if len(s) == 0:
        return np.array([])
    
    time_series = np.atleast_2d(s)
    step_ts = 1000 / output_fs
    tot_len = int((time_series[-1][0]-time_series[0][0]) / step_ts) + 1
    if tnew is None:
        xnew = time_series[0][0] + np.arange(0, tot_len*step_ts, step_ts)
    else:
        xnew = np.array(tnew)

    if verbose >= 1:
        print('time_series start ts = {}, end ts = {}'.format(time_series[0][0], time_series[-1][0]))
        print('tot_len = {}'.format(tot_len))
        print('xnew start = {}, end = {}'.format(xnew[0], xnew[-1]))

    if method == "spline":
        m = len(time_series)
        w = options.get("w", np.ones(shape=(m,)))
        # s = options.get("s", np.random.uniform(m-np.sqrt(2*m),m+np.sqrt(2*m)))
        s = options.get("s", m-np.sqrt(2*m))
        options.update(w=w, s=s)

        tck = interpolate.splrep(time_series[:,0],time_series[:,1],**options)

        regular_timeseries = interpolate.splev(xnew, tck)
    elif method == "interp1d":
        f = interpolate.interp1d(time_series[:,0],time_series[:,1],**options)

        regular_timeseries = f(xnew)
    
    if return_with_time:
        return np.column_stack((xnew, regular_timeseries))
    else:
        return regular_timeseries


def resample_discontinuous_irregular_timeseries(s:ArrayLike, allowd_gap:Optional[Real]=None,output_fs:Real=2, method:str="spline", return_with_time:bool=True, tnew:Optional[ArrayLike]=None, options:dict={}, verbose:int=0) -> List[np.ndarray]:
    """ finished, checked,

    resample the 2d discontinuous irregular timeseries `s` into a list of 1d or 2d regular time series with frequency `output_fs`,
    where discontinuity means time gap greater than `allowd_gap`,
    elements of `s` are in the form [time, value], where the unit of `time` is ms

    Parameters:
    -----------
    s: array_like,
        the 2d irregular timeseries
    output_fs: Real, default 2,
        the frequency of the output 1d regular timeseries
    method: str, default "spline"
        interpolation method, can be 'spline' or 'interp1d'
    return_with_time: bool, default False,
        return a 2d array, with the 0-th coordinate being time
    tnew: array_like, optional,
        the array of time of the output array
    options: dict, default {},
        additional options for the corresponding methods in scipy.interpolate

    Returns:
    --------
    list of np.ndarray, 1d or 2d regular time series with frequency `output_freq`

    NOTE:
    pandas also has the function to regularly resample irregular timeseries
    """
    time_series = np.atleast_2d(s)
    allowd_gap = allowd_gap or 2*1000/output_fs
    split_indices = [0] + (np.where(np.diff(time_series[:,0]) > allowd_gap)[0]+1).tolist() + [len(time_series)]
    if tnew is not None:
        l_tnew = [[p for p in tnew if time_series[split_indices[idx],0]<=p<time_series[split_indices[idx+1],0]] for idx in range(len(split_indices)-1)]
    else:
        l_tnew = [None for _ in range(len(split_indices)-1)]
    result = []
    for idx in range(len(split_indices)-1):
        r = resample_irregular_timeseries(
            s=time_series[split_indices[idx]: split_indices[idx+1]],
            output_fs=output_fs,
            method=method,
            return_with_time=return_with_time,
            tnew=l_tnew[idx],
            options=options,
            verbose=verbose
        )
        result.append(r)
    return result


def sft(s:ArrayLike) -> np.ndarray:
    """

    slow Fourier transform
    """
    N = len(s)
    _s = np.array(s)
    tmp = np.array(list(range(N)))
    return np.array([(_s*np.exp(-2*np.pi*1j*n*tmp/N)).sum() for n in range(N)])


def butter_bandpass(lowcut:Real, highcut:Real, fs:Real, order:int, verbose:int=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Butterworth Bandpass Filter Design

    Parameters:
    -----------
    lowcut: real,
        low cutoff frequency
    highcut: real,
        high cutoff frequency
    fs: real,
        frequency of `data`
    order: int,
        order of the filter
    verbose: int, default 0

    Returns:
    --------
    b, a: tuple of ndarray,
        coefficients of numerator and denominator of the filter

    NOTE:
    -----
    according to `lowcut` and `highcut`, the filter type might fall to lowpass or highpass filter

    References:
    -----------
    [2] scipy.signal.butter
    [1] https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    if low >= 1:
        raise ValueError("frequency out of range!")
    high = highcut / nyq

    if low <= 0 and high >= 1:
        b, a = [1], [1]
        return b, a
    
    if low <= 0:
        Wn = high
        btype = 'low'
    elif high >= 1:
        Wn = low
        btype = 'high'
    elif lowcut==highcut:
        Wn = high
        btype = 'low'
    else:
        Wn = [low, high]
        btype = 'band'
    
    if verbose >= 1:
        print('by the setup of lowcut and highcut, the filter type falls to {}, with Wn = {}'.format(btype, Wn))
    
    b, a = butter(order, Wn, btype=btype)
    return b, a


def butter_bandpass_filter(data:ArrayLike, lowcut:Real, highcut:Real, fs:Real, order:int, verbose:int=0) -> np.ndarray:
    """
    Butterworth Bandpass

    Parameters:
    -----------
    data: array_like,
        data to be filtered
    lowcut: real,
        low cutoff frequency
    highcut: real,
        high cutoff frequency
    fs: real,
        frequency of `data`
    order: int,
        order of the filter
    verbose: int, default 0

    Returns:
    --------
    y, ndarray,
        the filtered signal

    References:
    -----------
    [1] https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


@jit(nopython=True)
def hampel(input_series:ArrayLike, window_size:int, n_sigmas:int=3, return_outlier:bool=True) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    """
    """
    n = len(input_series)
    new_series = np.array(input_series).copy()
    k = 1.4826 # scale factor for Gaussian distribution
    outlier_indices = []
    
    for i in range((window_size),(n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            outlier_indices.append(i)
    if return_outlier:
        return new_series, outlier_indices
    else:
        return new_series


class MovingAverage(object):
    """

    moving average

    References:
    -----------
    [1] https://en.wikipedia.org/wiki/Moving_average
    """
    def __init__(self, data:ArrayLike, **kwargs):
        """
        """
        self.data = np.array(data)

        self.verbose = kwargs.get("verbose", 0)

    def cal(self, kind:str, **kwargs) -> np.ndarray:
        """
        """
        k = kind.lower().replace('_', ' ')
        if k in ['ema', 'ewma', 'exponential moving average', 'exponential weighted moving average']:
            func = self._ema
        elif k in ['cma', 'cumulative moving average']:
            func = self._cma
        elif k in ['wma', 'weighted moving average']:
            func = self._wma
        else:
            raise NotImplementedError
        return func(**kwargs)

    def _naive(self, **kwargs) -> np.ndarray:
        """
        """
        smoothed = []
        raise NotImplementedError

    def _ema(self, weight:float=0.6, **kwargs) -> np.ndarray:
        """
        """
        smoothed = []
        last = self.data[0]
        for d in self.data:
            s = last * weight + (1 - weight) * d
            last = s
            smoothed.append(s)
        return smoothed

    def _cma(self, **kwargs) -> np.ndarray:
        """
        """
        smoothed = []
        raise NotImplementedError

    def _wma(self, **kwargs) -> np.ndarray:
        """
        """
        smoothed = []
        raise NotImplementedError
