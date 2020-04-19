# -*- coding: utf-8 -*-
"""
utilities for visualization
"""
import numpy as np
from numbers import Real
from typing import Union, Optional, List, Tuple, NoReturn
import matplotlib.pyplot as plt

from ..common import ArrayLike


__all__ = [
    "plot_single_lead_ecg",
]


def plot_single_lead_ecg(s:ArrayLike, freq:Real, use_idx:bool=False, **kwargs) -> NoReturn:
    """ not finished

    single lead ECG plot,

    Parameters:
    -----------
    s: array_like,
        the single lead ECG signal
    freq: real,
        sampling frequency of `s`
    use_idx: bool, default False,
        use idx instead of time for the x-axis

    contributors: Jeethan, WEN Hao
    """
    default_fig_sz = 120
    line_len = freq * 25  # 25 seconds
    nb_lines, residue = divmod(len(s), line_len)
    if residue > 0:
        nb_lines += 1
    for idx in range(nb_lines):
        idx_start = idx*line_len
        idx_end = min((idx+1)*line_len, len(s))
        c = s[idx_start:idx_end]
        secs = np.arange(idx_start, idx_end)
        if not use_idx:
            secs = secs / freq
        mvs = np.array(c) * 0.001
        fig_sz = int(round(default_fig_sz * (idx_end-idx_start)/line_len))
        fig, ax = plt.subplots(figsize=(fig_sz, 6))
        ax.plot(secs, mvs, c='black')

        ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set_xlim(secs[0], secs[-1])
        ax.set_ylim(-1.5, 1.5)
        if use_idx:
            plt.xlabel('Samples')
        else:
            plt.xlabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        plt.show()
