# -*- coding: utf-8 -*-
"""
utilities for visualization
"""
import numpy as np
from numbers import Real
from typing import Union, Optional, List, Tuple, NoReturn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..common import ArrayLike


__all__ = [
    "plot_single_lead_ecg",
    "plot_hypnogram",
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


def plot_hypnogram(sleep_stage_curve:ArrayLike, style:str='original', **kwargs) -> NoReturn:
    """

    plot the hypnogram

    Parameters:
    -----------
    sleep_stage_curve: array_like,
        the sleep stage curve, each element is of the form 't, val',
        allowed stages are (case insensitive)
        - awake
        - REM
        - NREM1, NREM2, NREM3, NREM4
    style: str, default 'original'
        style of the hypnogram, can be the original style, or 'vspan'
    kwargs: dict,
        other key word arguments, including
        - ax: the axis to plot
    """
    all_stages = ['NREM4', 'NREM3', 'NREM2', 'NREM1', 'REM', 'awake',]
    all_stages = [item for item in all_stages if item.lower() in set([p[1].lower() for p in sleep_stage_curve])]
    all_stages = {all_stages[idx]:idx for idx in range(1,len(all_stages)+1)}

    palette = {
        'awake': 'orange',
        'REM': 'yellow',
        'NREM1': 'green',
        'NREM2': 'cyan',
        'NREM3': 'blue',
        'NREM4': 'purple',
    }
    patches = {k: mpatches.Patch(color=c, label=k) for k,c in palette.items()}

    ax = kwargs.get('ax', None)
    if ax is None:
        fig, ax = plt.subplots(figsize=(20,12))

    if style == 'original':
        pass
    elif style == 'vspan':
        pass

    raise NotImplementedError
