# -*- coding: utf-8 -*-
"""
utilities for visualization
"""
from numbers import Real
from typing import Union, Optional, List, Tuple, NoReturn
import matplotlib.pyplot as plt

from ..common import ArrayLike


__all__ = [
    "plot_single_lead_ecg",
]


def plot_single_lead_ecg(s:ArrayLike, freq:Real, l_waves:Optional[dict]=None, extra_info:Union[List[str],Tuple[str]]=['rr','pr','qrs','p','t'], extra_plot:Optional[List[dict]]=None, use_idx:bool=False, **kwargs) -> NoReturn:
    """ not finished

    single lead ECG plot,

    Parameters:
    -----------
    s: array_like,
        the single lead ECG signal
    freq: real,
        sampling frequency of `s`
    l_waves: dict,
        indices of the 'p', 'q', 'r', 's', 't' waves in `s`
    extra_info: list of str, default ['rr','pr','qrs','p','t'],
        extra information to print on the plot
    extra_plot: list of dict,
        of the form
        {
            'style': 'axvline',
            'position': [12,45],
            'content': ''
        }
    use_idx: bool, default False,
        use idx instead of time for the x-axis

    contributors: Jeethan, WEN Hao
    """
    l_ecg_beats = None
    default_fig_sz = 120
    line_len = freq * 25  # 25 seconds
    nb_lines = len(s) // line_len
    nb_beats_tot = 0
    for idx in range(nb_lines):
        if l_ecg_beats is not None:
            beats = [b for b in l_ecg_beats if idx*line_len<=b.r_peak_idx_abs<(idx+1)*line_len]
        else:
            beats = []
        c = s[idx*line_len:(idx+1)*line_len]
        secs = np.arange(idx*line_len, (idx+1)*line_len)
        if not use_idx:
            secs = secs / freq
        mvs = np.array(c) * 0.001
        fig_sz = default_fig_sz
        fig, ax = plt.subplots(figsize=(fig_sz,6))
        ax.plot(secs, mvs, c='black')
        # plot extra_info
        for b_idx, b in enumerate(beats):
            # x_pos = min(b.r_peak_idx_abs-idx*line_len+5, line_len)
            r_pos = min(b.r_peak_idx_abs+5, (idx+1)*line_len)/freq
            y_pos = min(1.3, b.beat[b.r_peak_idx]/1000+0.1)
            ax.text(r_pos, y_pos, str(b_idx+nb_beats_tot), color='black', fontsize=12)
            if 'rr' in extra_info and b.rr_next > 0:
                x_pos = min(r_pos+b.rr_next/2500, (idx+1)*line_len/freq)
                y_pos = min(1.1, 0.85*b.beat[b.r_peak_idx]/1000)
                ax.text(x_pos, y_pos, str(b.rr_next), color='green', fontsize=12)
            if 'pr' in extra_info and b.pr > 0:
                x_pos = max(r_pos-b.pr/1000, idx*line_len/freq)
                y_pos = min(0.5, 0.35*b.beat[b.r_peak_idx]/1000)
                ax.text(x_pos, y_pos, str(b.pr), color='green', fontsize=12)
            if 'qrs' in extra_info and b.qrs_onset_idx>DEFAULT_IDX and b.qrs_offset_idx>DEFAULT_IDX:
                x_pos = max(r_pos-0.05, idx*line_len/freq)
                y_pos = max(-1.4, min(-0.3, np.min(b.beat[b.qrs_onset_idx:b.qrs_offset_idx])/1000-0.1))
                ax.text(x_pos, y_pos, str(b.qrs_width), color='green', fontsize=12)
            if 'p' in extra_info and b.p_peak_idx>DEFAULT_IDX:
                pass
            if 't' in extra_info and b.t_peak_idx>DEFAULT_IDX:
                pass
        # TODO: plot extra_plot

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
        plt.ylabel('Voltage (mV)')
        plt.show()
        nb_beats_tot += len(beats)
    
    c = s[nb_lines*line_len:]  # tail left
    if len(c) > 0:
        if l_ecg_beats is not None:
            beats = [b for b in l_ecg_beats if nb_lines*line_len<=b.r_peak_idx_abs]
        else:
            beats = []
        secs = np.arange(nb_lines*line_len, len(s))
        if not use_idx:
            secs = secs / freq
        mvs = np.array(c) * 0.001
        fig_sz = int(default_fig_sz*(len(s)-nb_lines*line_len)/line_len)
        fig, ax = plt.subplots(figsize=(fig_sz,6))
        ax.plot(secs, mvs, c='black')
        # plot extra_info
        for b_idx, b in enumerate(beats):
            r_pos =min(b.r_peak_idx_abs+5, len(s))/freq
            y_pos = min(1.3, b.beat[b.r_peak_idx]/1000+0.1)
            ax.text(r_pos, y_pos, str(b_idx+nb_beats_tot), color='black', fontsize=12)
            if 'rr' in extra_info and b.rr_next > 0:
                x_pos = min(r_pos+b.rr_next/2500, len(s)/freq)
                y_pos = min(1.1, 0.85*b.beat[b.r_peak_idx]/1000)
                ax.text(x_pos, y_pos, str(b.rr_next), color='green', fontsize=12)
            if 'pr' in extra_info and b.pr > 0:
                x_pos = max(r_pos-b.pr/1000, nb_lines*line_len/freq)
                y_pos = min(0.5, 0.35*b.beat[b.r_peak_idx]/1000)
                ax.text(x_pos, y_pos, str(b.pr), color='green', fontsize=12)
            if 'qrs' in extra_info and b.qrs_onset_idx>DEFAULT_IDX and b.qrs_offset_idx>DEFAULT_IDX:
                x_pos = max(r_pos-0.05, nb_lines*line_len/freq)
                y_pos = max(-1.4, min(-0.3, np.min(b.beat[b.qrs_onset_idx:b.qrs_offset_idx])/1000-0.1))
                ax.text(x_pos, y_pos, str(b.qrs_width), color='green', fontsize=12)
            if 'p' in extra_info and b.p_peak_idx>DEFAULT_IDX:
                pass
            if 't' in extra_info and b.t_peak_idx>DEFAULT_IDX:
                pass
        # TODO: plot extra_info
        
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
        plt.ylabel('Voltage (mV)')
        plt.show()
