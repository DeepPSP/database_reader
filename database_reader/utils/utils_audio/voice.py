# -*- coding: utf-8 -*-
'''
use various backends for audio reading, features extracting, etc.

About the python backends:
--------------------------
wave:
    the native python module,
    only has basic audio io operations
librosa:
    https://github.com/librosa/librosa
praat(parselmouth):
    https://github.com/YannickJadoul/Parselmouth
    http://www.fon.hum.uva.nl/praat/
pyAudioAnalysis:
    https://github.com/tyiannak/pyAudioAnalysis
pydub:
    https://github.com/jiaaro/pydub
SoundFile:
    https://github.com/bastibe/SoundFile
audioread
    https://github.com/beetbox/audioread

other packages；
    https://github.com/tuffy/python-audio-tools (or http://audiotools.sourceforge.net)

Libraries behind the python packages:
FFmpeg, libsndfile, praat, MAD,

installation for centos7:
https://gist.github.com/wenh06/de3f1a35b242df8059ce7c24e4c1784c
'''
import wave
import librosa
import pydub
import parselmouth as pm
import soundfile as sf
import pyAudioAnalysis as paa
import os
import numpy as np
from collections import namedtuple
from numbers import Real
from typing import Union, Optional, List, NoReturn
from parselmouth.praat import call
from scipy.interpolate import interp1d

from utils import ArrayLike
from utils.utils_signal import butter_bandpass_filter, MovingAverage
from utils.utils_universal import generalized_intervals_intersection
from utils.utils_misc import indicator_enter_leave_func


_DEFAULT_BACKEND = 'praat'
_VERBOSE_LEVEL = 0


__all__ = [
    "Voice",
    "VoiceVowel",
    "SyllableSegment",
]


class Voice(object):
    """
    class for analyzing human voice

    TODO:
    1. automatic switching backends for robust reading of audio files
    2. better plot
    3. more voice features
    """
    def __init__(self, values:Optional[np.ndarray]=None, freq:Optional[Real]=None, start_time:Optional[Real]=None, l_file_path:Optional[List[str]]=None, **kwargs):
        """
        Parameter:
        ----------
        values: ndarray, optional,
            values of the audio recording
        freq: real, optional,
            sampling frequency of the audio recording
        start_time: real, optional,
            start time of the audio recording, units in seconds
        l_file_path: list of str, optional,
            list of the paths of the audio files,
            voices in these files will be concatenated
        """
        self.values = values
        self.filtered_values = None
        self.freq = freq
        self.dt = 1/freq if freq is not None else None  # time spacing
        self.duration = len(self.values)/self.freq if self.freq is not None else None
        self.start_time = start_time
        self.l_file_path = l_file_path
        self.kwargs = kwargs

        if all([self.values is not None, self.freq is not None]):
            self._loaded = True
        else:
            self._loaded = False

        self.verbose = kwargs.get("verbose", 0)
        self.be_praat, self.be_praat_params = None, None
        
        self._pitches = None
        self._formants = None
        self._harmonicity = None
        self._spectrogram = None
        self._spectrum = None
        self._intensity = None
        self._mfcc = None
        self._jitter = None

        self.vowels = []
        self.syllable_segments = []
        self.d_vuv = {}
        self.vot = []
        self.f2 = []
        self.f2_slope = []


    def reset(self) -> NoReturn:
        """
        """
        self._pitches = None
        self._formants = None
        self._harmonicity = None
        self._spectrogram = None
        self._spectrum = None
        self._intensity = None
        self._mfcc = None
        self._jitter = None

        self.vowels = []
        self.syllable_segments = []
        self.d_vuv = {}
        self.vot = []
        self.f2 = []
        self.f2_slope = []
    

    def load(self, backend:str='librosa', **kwargs) -> NoReturn:
        """ partly finished,

        Parameters:
        -----------
        backend: str, default `librosa',
            backend module that reads the audio files
        kwargs: dict,
            arguments for the `backend` reading the audio files,
            for example `sr` for `librosa`

        NOTE:
        1. files that Praat can read:
        http://www.fon.hum.uva.nl/praat/manual/Sound_files_3__Files_that_Praat_can_read.html
        """
        if self._loaded:
            return
        if self.l_file_path is None:
            raise ValueError("No data to load!")
        
        self.be_praat = None
        self.be_wave = None
        
        if backend.lower() == 'praat':
            self.values = np.array([])
            for file_path in self.l_file_path:
                be_praat_tmp = pm.Sound(file_path, **kwargs) # TODO: 多个文件拼接
                self.values = np.append(self.values, be_praat_tmp.values[0], axis=0)
                self.freq = be_praat_tmp.sampling_frequency
            self.be_praat = pm.Sound(
                values=self.values,
                sampling_frequency=self.freq,
            )
            self.dt = self.be_praat.dt
            self.duration = len(self.values) * self.dt
            self._loaded = True
        elif backend.lower() == 'librosa':
            for file_path in self.l_file_path:
                tmp_values, tmp_freq = librosa.load(path=file_path, **kwargs)
                self.values = np.append(self.values, tmp_values, axis=0) if self.values is not None else tmp_values
                if self.freq is None:
                    self.freq = tmp_freq
                else:
                    assert self.freq == tmp_freq
            self.duration = len(self.values)/self.freq
            self.dt = 1/self.freq
            self._loaded = True
        elif backend.lower() == 'soundfile':
            self.values, self.freq = sf.read(file=self.l_file_path[0], **kwargs)# TODO: 多个文件拼接
            self._loaded = True
        elif backend.lower() == 'wave':
            self.be_wave = wave
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.verbose >= 1:
            print("len(self.values) = {}, self.freq = {}".format(len(self.values), self.freq))


    def resample(self, new_freq:Real) -> NoReturn:
        """

        Paramters:
        ----------
        new_freq: real,
            the new frequency
        """
        if not self._loaded:
            self.load(backend='librosa', sr=new_freq)
            return
        
        self.values = librosa.resample(
            self.values,
            orig_sr=self.freq, target_sr=new_freq,
            res_type='kaiser_best', fix=True, scale=False,
        )
        if self.filtered_values is not None:
            self.bandpass_filter()
        self.freq = new_freq
        self.dt = 1/self.freq
        self.reset()


    def save(self, filename:str, fmt:Optional[str]='wav', freq:Optional[Real]=None, **kwargs):
        """

        Parameters:
        -----------
        filename: str,
            name of the file to save
        fmt: str, default 'wav', optional,
            format of the file to save
        freq: real, optional,
            sampling frequency of the file to save
        """
        fmt = os.path.splitext(filename)[1].replace('.', '') or fmt
        if not self._loaded:
            self.load(backend='librosa', sr=freq)
            freq = self.freq
            to_save_values = self.values
        elif freq is not None:
            to_save_values = librosa.resample(
                self.values,
                orig_sr=self.freq, target_sr=freq,
                res_type='kaiser_best', fix=True, scale=False,
            )
        else:
            freq = self.freq
            to_save_values = self.values
        sf.write(filename, to_save_values, freq, format=fmt, **kwargs)


    def preprocess(self,):
        """ not finished,
        """
        raise NotImplementedError


    def bandpass_filter(self, **kwargs) -> NoReturn:
        """ finished,

        Parameters:
        -----------
        kwargs: dict,
            arguments for butter bandpass filter, including 'lowcut', 'highcut', 'order'
        """
        self.filtered_values = butter_bandpass_filter(
            data=self.values,
            lowcut=kwargs.get('lowcut', 80),
            highcut=kwargs.get('highcut', 8000),
            fs=self.freq,
            order=kwargs.get('order', 3)
        )


    def analyze(self, items:Optional[List[str]]=None) -> NoReturn:
        """ not finished,
        """
        raise NotImplementedError


# -------------------------------------------------------------
# low level features

    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_intensity(self, backend:str='praat', time_range:Optional[ArrayLike]=None, is_filtered:bool=False, **kwargs) -> NoReturn:
        """ partly finished,

        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation
        is_filtered: bool, default False,
            if True, the (butter bandpass) filtered voice values will be used for computation

        References:
        -----------
        [1] https://en.wikipedia.org/wiki/Sound_intensity
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_intensity  "+"*"*10+"\n")
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            if not is_filtered:
                snd = pm.Sound(
                    values=self.values[st_idx:ed_idx],
                    sampling_frequency=self.freq,
                )
            else:
                if self.filtered_values is None:
                    self.bandpass_filter()
                snd = pm.Sound(
                    values=self.filtered_values[st_idx:ed_idx],
                    sampling_frequency=self.freq,
                )
            intensity = snd.to_intensity()
            self._intensity = {}
            self._intensity['ts'] = intensity.ts()
            self._intensity['values'] = intensity.values.flatten()
        else:
            raise NotImplementedError
    
    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_pitches(self, backend:str='praat', time_range:Optional[ArrayLike]=None, **kwargs) -> NoReturn:
        """ partly finished,

        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_pitches  "+"*"*10+"\n")
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            snd = pm.Sound(
                values=self.values[st_idx:ed_idx],
                sampling_frequency=self.freq,
            )
            pitches = snd.to_pitch()
            self._pitches = {}
            self._pitches['xs'] = pitches.xs()
            self._pitches['frequency'] = pitches.selected_array['frequency']
        else:
            raise NotImplementedError
    
    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_formants(self, backend:str='praat', time_range:Optional[ArrayLike]=None, **kwargs) -> NoReturn:
        """ partly finished,

        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_formants  "+"*"*10+"\n")
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            snd = pm.Sound(
                values=self.values[st_idx:ed_idx],
                sampling_frequency=self.freq,
            )
            formants = snd.to_formant_burg()
            x = formants.xs()
            nb_x = formants.nx
            maximum_formant = kwargs.get("maximum_formant", 5)
            self._formants = {}
            self._formants['xs'] = x
            for fn in range(1,maximum_formant+1):
                self._formants['F'+str(fn)] = np.array([formants.get_value_at_time(fn, x[idx]) for idx in range(nb_x)])
        else:
            raise NotImplementedError

    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_formants_with_power(self, backend:str='praat', time_range:Optional[ArrayLike]=None, **kwargs) -> NoReturn:
        """ partly finished,

        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_formants  "+"*"*10+"\n")
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            snd = pm.Sound(
                values=self.values[st_idx:ed_idx],
                sampling_frequency=self.freq,
            )
            formants = snd.to_formant_burg()
            spectrogram = snd.to_spectrogram()
            x = formants.xs()
            nb_x = formants.nx
            maximum_formant = kwargs.get("maximum_formant", 5)
            self._formants = {}
            self._formants['xs'] = x
            for fn in range(1, maximum_formant+1):
                self._formants['F'+str(fn)] = []
                self._formants['F'+str(fn)+'_power'] = []
                for idx in range(nb_x):
                    fs = formants.get_value_at_time(fn, x[idx])
                    if np.isnan(fs):
                        pw = np.nan
                    else:
                        pw = spectrogram.get_power_at(x[idx], fs)
                    self._formants['F'+str(fn)].append(fs)
                    self._formants['F'+str(fn)+'_power'].append(pw)
                self._formants['F'+str(fn)] = np.array(self._formants['F'+str(fn)])
                self._formants['F'+str(fn)+'_power'] = np.array(self._formants['F'+str(fn)+'_power'])
            # self._formants['F'+str(fn)] = np.array([ for idx in range(nb_x)])
            # self._formants['F'+str(fn)+'_power'] = np.array([spectrogram.get_power_at(x[idx], formants.get_value_at_time(fn, x[idx])) for idx in range(nb_x)])
        else:
            raise NotImplementedError

    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_harmonicity(self, backend:str='praat', time_range:Optional[ArrayLike]=None, **kwargs) -> NoReturn:
        """ partly finished,

        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_harmonicity  "+"*"*10+"\n")
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            snd = pm.Sound(
                values=self.values[st_idx:ed_idx],
                sampling_frequency=self.freq,
            )
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            self._harmonicity = {}
            self._harmonicity['hnr'] = hnr
        else:
            raise NotImplementedError
    
    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_spectrogram(self, backend:str='praat', time_range:Optional[ArrayLike]=None, **kwargs) -> NoReturn:
        """ not finished,

        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_spectrogram  "+"*"*10+"\n")
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            snd = pm.Sound(
                values=self.values[st_idx:ed_idx],
                sampling_frequency=self.freq,
            )
            self._spectrogram = snd.to_spectrogram()
        else:
            raise NotImplementedError

    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_spectrum(self, backend:str='praat', time_range:Optional[ArrayLike]=None, **kwargs) -> NoReturn:
        """ not finished,

        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_spectrum  "+"*"*10+"\n")
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            snd = pm.Sound(
                values=self.values[st_idx:ed_idx],
                sampling_frequency=self.freq,
            )
            self._spectrum = snd.to_spectrum()
        else:
            raise NotImplementedError

    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_mfcc(self, backend:str='praat', time_range:Optional[ArrayLike]=None, **kwargs) -> NoReturn:
        """ not finished,

        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_mfcc  "+"*"*10+"\n")
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            snd = pm.Sound(
                values=self.values[st_idx:ed_idx],
                sampling_frequency=self.freq,
            )
            self._mfcc = snd.to_mfcc()
        else:
            raise NotImplementedError

    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_jitter(self, backend:str='praat', time_range:Optional[ArrayLike]=None, **kwargs) -> NoReturn:
        """ partly finished,
        
        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_jitter  "+"*"*10+"\n")
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            snd = pm.Sound(values=self.values[st_idx:ed_idx], sampling_frequency=self.freq)
            f0min, f0max = kwargs.get("f0min", 75), kwargs.get("f0max", 500)
            pointProcess = call(snd, "To PointProcess (periodic, cc)", f0min, f0max)
            localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            self._jitter = {}
            self._jitter['local'] = localJitter
            self._jitter['local_abs'] = localabsoluteJitter
            self._jitter['rap'] = rapJitter
            self._jitter['ppq5'] = ppq5Jitter
            self._jitter['ddp'] = ddpJitter
                
        else:
            raise NotImplementedError


    def equalize(self, freqs:List[ArrayLike], gains:Union[List[Real],Real], inplace:bool=False, **kwargs):
        """

        Parameters:
        -----------
        freqs: list of array_like,
            list of frequency bands to be equalized
        gains: real, or list of real,
            the gains corresponding to the `freqs` for equalization
        inplace: bool, default False

        Returns:
        --------
        """
        se = self.values.copy()
        if isinstance(gains, Real):
            gains = [gains for _ in freqs]
        for idx, (low, high) in enumerate(freqs):
            g = gains[idx]
            se += butter_bandpass_filter(self.values, low, high, self.freq, order=kwargs.get("order", 3), verbose=self.verbose)*np.power(10, g/20)
        if inplace:
            self.reset()
            self.values = se
        else:
            return Voice(values=se, freq=self.freq, start_time=self.start_time, l_file_path=self.l_file_path, **self.kwargs)


    def energy_proportion_curve(self, lowcut:Real, highcut:Real, time_range:Optional[ArrayLike]=None, smoothing:float=0.6, **kwargs) -> np.ndarray:
        """

        Parameters:
        -----------
        lowcut: real,
            lower bound of the frequency band to compute energy proportion
        highcut: real,
            higher bound of the frequency band to compute energy proportion
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        """
        assert lowcut < highcut, "Invalid frequency range"
        df = np.diff(self.spectrogram.y_grid())[0]
        assert df < highcut - lowcut, "frequency range should be larger than the resolution of frequencies in the spectrogram"
        ts = self.spectrogram.x_grid()[:-1]
        spectrogram = self.spectrogram.values
        if time_range is not None:
            t_indices = np.where((ts>time_range[0])&(ts<time_range[1]))[0]
            ts = ts[t_indices]
            spectrogram = spectrogram[:, t_indices]
        frequencies = self.spectrogram.y_grid()[1:]
        tot_energy = np.nansum(spectrogram, axis=0)
        f_indices = np.where((frequencies>=lowcut)&(frequencies<=highcut))[0]
        band_spectrogram = spectrogram[f_indices, :]
        band_energy = np.nansum(band_spectrogram, axis=0)
        epc = band_energy / (tot_energy + np.finfo(float).eps)
        ma = MovingAverage(epc)
        epc = ma.cal(kind='exponential moving average', weight=smoothing)
        epc = np.column_stack((ts, epc))
        return epc


    def energy_proportion(self, lowcut:Real, highcut:Real, time_range:Optional[ArrayLike]=None, **kwargs) -> float:
        """

        Parameters:
        -----------
        lowcut: real,
            lower bound of the frequency band to compute energy proportion
        highcut: real,
            higher bound of the frequency band to compute energy proportion
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation

        Returns:
        --------
        """
        epc = self.energy_proportion_curve(lowcut=lowcut, highcut=highcut, time_range=time_range, **kwargs)
        propotion = np.nanmean(epc[:,1])
        return propotion


# -------------------------------------------------------------
# syllable level features

    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_syllable_segments(self, backend:str='praat', time_range:Optional[ArrayLike]=None, intensity_threshold:Real=40, t_threshold:Real=0.06) -> NoReturn:
        """ partly finished,

        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation
        intensity_threshold: real, default 40,
        t_threshold: real, default 0.06,

        Returns:
        --------
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*10+"  entering function obtain_syllable_segments  "+"*"*10+"\n")
        if backend.lower() == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            analysis_start_time = time_range[0] if time_range is not None else self.be_praat.start_time
            snd = pm.Sound(
                values=self.values[st_idx:ed_idx],
                sampling_frequency=self.freq,
                start_time=analysis_start_time,
            )
            intensity = snd.to_intensity()
            t_arr = intensity.ts()
            intensity_values = intensity.values[0].flatten()
        else:
            raise NotImplementedError

        voice_indices = np.where(intensity_values > intensity_threshold)[0]
        split_indices = [0] + (np.where(np.diff(voice_indices)>1)[0]+1).tolist() + [len(voice_indices)]
            
        if self.verbose >= 1:
            print("voice_indices = {}".format(voice_indices))
            print("split_indices = {}".format(split_indices))
            
        self.syllable_segments = []
        for idx in range(len(split_indices)-1):
            start_idx = voice_indices[split_indices[idx]]
            end_idx = voice_indices[split_indices[idx+1]-1]
            start_t = t_arr[start_idx]
            end_t = t_arr[end_idx]
            if self.verbose >= 2:
                print("at the {}-th candidate segment,".format(idx))
                print("start_idx = {}, end_idx = {}".format(start_idx, end_idx))
                print("start_t = {}, end_t = {}, diff = {}".format(start_t, end_t, end_t-start_t))
            if end_t - start_t <= t_threshold:
                continue
            ori_indices = np.where((start_t<=self.ts())&(end_t>=self.ts()))[0]
            self.syllable_segments.append(
                SyllableSegment(
                    values=self.values.flatten()[ori_indices], freq=self.freq, start_time=start_t, end_time=end_t
                )
            )

    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_vowels(self, backend:str='praat', time_range:Optional[ArrayLike]=None, trim_by_syllable:bool=True, **kwargs) -> NoReturn:
        """ partly finished,
        
        Parameters:
        -----------
        backend: str, default 'praat',
            backend for computation of voice intensity
        time_range: array_like, optional,
            of the form [start_sec, end_sec], time range for computation
        trim_by_syllable: bool, default True,

        Returns:
        --------
        """
        # if self.verbose >= 1:
        #     print("\n"+"*"*15+"  entering function obtain_vowels  "+"*"*15+"\n")
        if trim_by_syllable and len(self.syllable_segments) == 0:
            self.obtain_syllable_segments(
                backend=backend,
                time_range=time_range,
            )
        if backend == 'praat':
            if time_range is None:
                st_idx, ed_idx = 0, len(self.values)
            else:
                st_idx, ed_idx = int(time_range[0] * self.freq), int(time_range[1] * self.freq)
            if st_idx >= ed_idx:
                if self.verbose >= 1:
                    print("st_idx () is greater than or equal to ed_idx, empty value")
                return
            snd = pm.Sound(values=self.values[st_idx:ed_idx], sampling_frequency=self.freq)
            pitches = snd.to_pitch()
            t_arr = pitches.xs()
            pitch_values = pitches.selected_array['frequency']
        else:
            raise NotImplementedError
            
        vowel_indices = np.where(pitch_values > 0)[0]
        split_indices = [0] + (np.where(np.diff(vowel_indices)>1)[0]+1).tolist() + [len(vowel_indices)]
        if trim_by_syllable:
            syllable_time_ranges = [[item.start_time, item.end_time] for item in self.syllable_segments]
        else:
            syllable_time_ranges = []

        if self.verbose >= 1:
            print("vowel_indices = {}".format(vowel_indices.tolist()))
            print("split_indices = {}".format(split_indices))
            print("syllable_time_ranges = {}".format(syllable_time_ranges))
            
        self.vowels = []
        for idx in range(len(split_indices)-1):
            start_idx = vowel_indices[split_indices[idx]]
            end_idx = vowel_indices[split_indices[idx+1]-1]
            start_t = t_arr[start_idx]
            end_t = t_arr[end_idx]
            if self.verbose >= 2:
                print("at the {}-th candidate segment,".format(idx))
                print("start_idx = {}, end_idx = {}".format(start_idx, end_idx))
                print("start_t = {}, end_t = {}, diff = {}".format(start_t, end_t, end_t-start_t))
            if trim_by_syllable:
                trimed_time_range = generalized_intervals_intersection(
                    generalized_interval=[[start_t, end_t]],
                    another_generalized_interval=syllable_time_ranges,
                    drop_degenerate=True
                )
            else:
                trimed_time_range = [[start_t, end_t]]
            if len(trimed_time_range) > 0:
                start_t, end_t = trimed_time_range[0]
                trimed_indices = np.where((start_t<=t_arr)&(t_arr<=end_t))[0]
            else:
                start_t, end_t = -1, -1
            if self.verbose >= 2:
                print("after trimed by syllable_time_ranges, start_t = {}, end_t = {}, diff = {}".format(start_t, end_t, end_t-start_t))
            if start_t >= 0:
                self.vowels.append(
                    VoiceVowel(
                        start_time=start_t,
                        end_time=end_t,
                        frequencies=pitch_values[start_idx:end_idx+1],
                        ts=t_arr[start_idx:end_idx+1]
                    )
                )


# ------------------------------------------------------------
# higher level features

    @indicator_enter_leave_func(verbose=_VERBOSE_LEVEL)
    def obtain_vuv(self, **kwargs):
        """

        Parameters:
        -----------

        Returns:
        --------
        """
        raise NotImplementedError


# ------------------------------------------------------------
# plot

    def plot(self, items:Optional[List[str]]=None, **kwargs) -> NoReturn:
        """

        Parameters:
        -----------

        TODO: add time_range
        """
        font_prop = kwargs.get("font_prop", None)
        import matplotlib.pyplot as plt
        if items is None:
            items = ['signal']
        elif items == 'all':
            items = ['spectrogram', 'signal', 'pitches', 'formants', 'intensity', 'vuv']

        if isinstance(items, str):
            items = [items.lower()]
        else:
            items = [i.lower() for i in items]

        if 'praat' in items:
            self._plot_praat(**kwargs)
            return

        if 'signal' in items:
            fig, ax = plt.subplots(figsize=(20,4))
            ax.plot(np.arange(len(self.values))/self.freq, self.values, '-')
            ax.grid()
            ax.set_title('signal')
            ax.set_xlabel("time [s]")
            plt.show()
        if 'pitches' in items:
            if self._pitches is None:
                self.obtain_pitches(backend='praat')
            fig, ax = plt.subplots(figsize=(20,4))
            ax.plot(self._pitches['xs'], self._pitches['frequency'], '-')
            ax.grid()
            ax.set_title('pitches')
            ax.set_xlabel("time [s]")
            ax.set_ylabel("fundamental frequency [Hz]")
            plt.show()
        if 'formants' in items:
            if self._formants is None:
                self.obtain_formants(backend='praat')
            maximum_formant = kwargs.get("maximum_formant", 5)
            fig, ax = plt.subplots(figsize=(20,4))
            for fn in range(1,maximum_formant+1):
                ax.plot(self._formants['xs'], self._formants['F'+str(fn)], '-', label='F'+str(fn))
            ax.grid()
            ax.legend()
            ax.set_title('formants')
            ax.set_xlabel("time [s]")
            ax.set_ylabel("frequency [Hz]")
            plt.show()
        if 'intensity' in items:
            if self._intensity is None:
                self.obtain_intensity(backend='praat')
            fig, ax = plt.subplots(figsize=(20,4))
            ax.plot(self._intensity['ts'], self._intensity['values'], '-')
            ax.grid()
            ax.set_title('intensity')
            ax.set_xlabel("time [s]")
            ax.set_ylabel("intensity [dB]")
            plt.show()
        if 'vuv' in items:
            if self.d_vuv == {}:
                self.obtain_vuv()
            fig, ax = plt.subplots(figsize=(20,4))
            ax.plot(np.arange(len(self.values))/self.freq, self.values, '-')
            pause_seg = self.d_vuv['pause']
            for r in pause_seg:
                ax.axvspan(r[0], r[1], color='red', alpha=0.2)
            ax.grid()
            ax.set_xlabel("time [s]")
            ax.set_title('vuv')
            plt.show()
        if 'spectrogram' in items:
            fig, ax = plt.subplots(figsize=(20,4))
            self._plot_spectrogram(
                ax=ax,
                cmap=kwargs.get("cmap", plt.get_cmap("afmhot")),
                dynamic_range=70
            )
            if self._formants is None:
                self.obtain_formants(backend='praat')
            maximum_formant = kwargs.get("maximum_formant", 5)
            for fn in range(1,maximum_formant+1):
                ax.plot(self._formants['xs'], self._formants['F'+str(fn)], '-', label='F'+str(fn))
            # ax2 = ax.twinx()
            # ax2.plot(self._pitches['xs'], self._pitches['frequency'], '-')
            # ax2.set_ylabel("fundamental frequency [Hz]")
            ax.legend()
            ax.set_title('spectrogram')
            plt.show()

    
    def _plot_praat(self, **kwargs):
        """
        Parameters:
        -----------

        """
        if 'plt' not in dir():
            import matplotlib.pyplot as plt
        silence_threshold = kwargs.get('silence_threshold', 0.003)
        
        snd = pm.Sound(
                values=self.values,
                sampling_frequency=self.freq,
            )
        
        fig, (ax_t, ax_f) = plt.subplots(2,1,figsize=(int(25*snd.xmax),10),sharex=True)
        plt.subplots_adjust(hspace=0)
        ax_t.plot(snd.xs(), snd.values.T)
        ax_t.axhline(silence_threshold, linestyle='dashed', linewidth=0.5, color='red')
        ax_t.axhline(-silence_threshold, linestyle='dashed', linewidth=0.5, color='red')

        # plot intensity
        ax_t2 = ax_t.twinx()
        snd_intensity = snd.to_intensity()
        ax_t2.plot(snd_intensity.xs(), snd_intensity.values.T, 'o-', markersize=4, linewidth=0.6, color='yellow')
        if len(self.syllable_segments) == 0:
            self.obtain_syllable_segments()
        for seg in self.syllable_segments:
            ax_t2.axvspan(seg.start_time, seg.end_time, color='green', alpha=0.3)
        ax_t2.grid(False)
        ax_t2.set_ylim(0)
        ax_t2.set_ylabel("intensity [dB]")
        ax_t2.set_xlim([snd.xmin, snd.xmax])

        # plot spectrogram
        dynamic_range = kwargs.get('dynamic_range', 70)
        snd_spectrogram = snd.to_spectrogram()
        X, Y = snd_spectrogram.x_grid(), snd_spectrogram.y_grid()
        sg_db = 10 * np.log10(snd_spectrogram.values)
        cm = kwargs.get("cmap", plt.get_cmap("afmhot"))
        ax_f.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap=cm)
        ax_f.set_ylim([snd_spectrogram.ymin, snd_spectrogram.ymax])
        ax_f.set_xlabel("time [s]")
        ax_f.set_ylabel("frequency [Hz]")

        # plot pitches
        ax_f2 = ax_f.twinx()
        snd_pitches = snd.to_pitch()
        pitch_values = snd_pitches.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
        ax_f2.plot(snd_pitches.xs(), pitch_values, 'o-', markersize=9, color='w')
        ax_f2.plot(snd_pitches.xs(), pitch_values, 'o-', markersize=5, color='b')
        ax_f2.grid(False)
        ax_f2.set_ylim(0, snd_pitches.ceiling)
        ax_f2.set_ylabel("fundamental frequency [Hz]")
        ax_f2.set_xlim([snd.xmin, snd.xmax])

        # plot vowels
        if len(self.vowels) == 0:
            self.obtain_vowels()
        for v in self.vowels:
            ax_t.axvspan(v.start_time, v.end_time, color='red', alpha=0.5)

        # plot formants
        maximum_formant = kwargs.get("maximum_formant", 5)
        snd_formants = snd.to_formant_burg()
        x = snd_formants.xs()
        nb_x = snd_formants.nx
        for fn in range(1,maximum_formant+1):
            y = np.array([snd_formants.get_value_at_time(fn, x[idx]) for idx in range(nb_x)])
            ax_f.plot(x, y, 'o', markersize=5, color='w')
            ax_f.plot(x, y, 'o', markersize=2, color='r')

        plt.show()


    def _plot_spectrogram(self, ax, cmap, dynamic_range=70, **kwargs):
        """
        """
        font_prop = kwargs.get("font_prop", None)
        X, Y = self.spectrogram.x_grid(), self.spectrogram.y_grid()
        sg_db = 10 * np.log10(self.spectrogram.values)
        ax.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap=cmap)
        ax.set_ylim([self.spectrogram.ymin, self.spectrogram.ymax])
        ax.set_xlabel("time [s]")
        ax.set_ylabel("frequency [Hz]")


    def _plot_energy_proportion(self, lowcut:Real, highcut:Real, **kwargs):
        """
        """
        c = self.energy_proportion_curve(lowcut=lowcut, highcut=highcut, **kwargs)
        ts = c[:,0]
        proportions = c[:,1]

        import matplotlib.pyplot as plt
        font_prop = kwargs.get("font_prop", None)
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(ts, proportions, '-')
        ax.grid()
        ax.set_title('energy proportion curve')
        ax.set_xlabel("time [s]")
        ax.set_ylabel("propotion [%]")
        plt.show()

# ------------------------------------------------------------
# properties of lower level feature

    def ts(self) -> np.ndarray:
        """ follow the method ts() of a parselmouth.Sound object
        """
        return np.arange(0.5, 0.5 + len(self.values), 1) / self.freq

    @property
    def intensity(self):
        if self._intensity is None:
            self.obtain_intensity()
        return self._intensity

    @property
    def pitches(self):
        if self._pitches is None:
            self.obtain_pitches()
        return self._pitches

    @property
    def formants(self):
        if self._formants is None:
            self.obtain_formants()
        return self._formants

    @property
    def harmonicity(self):
        if self._harmonicity is None:
            self.obtain_harmonicity()
        return self._harmonicity

    @property
    def spectrogram(self):
        if self._spectrogram is None:
            self.obtain_spectrogram()
        return self._spectrogram

    @property
    def spectrum(self):
        if self._spectrum is None:
            self.obtain_spectrum()
        return self._spectrum

    @property
    def mfcc(self):
        if self._mfcc is None:
            self.obtain_mfcc()
        return self._mfcc

    @property
    def jitter(self):
        if self._jitter is None:
            self.obtain_jitter()
        return self._jitter


class VoiceVowel(object):
    """
    vowel
    """
    def __init__(self, start_time:Real, end_time:Real, frequencies:np.ndarray, ts:np.ndarray):
        """
        """
        self.start_time = start_time
        self.end_time = end_time
        self.frequencies = frequencies
        self._ts = ts
    
    def ts(self) -> np.ndarray:
        """ follow the method ts() of a parselmouth.Sound object
        """
        return np.array(self._ts)
        
    @property
    def duration(self):
        """
        """
        return self.start_time - self.end_time
    

class SyllableSegment(Voice):
    """
    syllable segment
    """
    def __init__(self, values: np.ndarray, freq:Real, start_time:Real, end_time:Real, vowel:Optional[Real]=None):
        """
        """
        super().__init__(values, freq, start_time)
        self.end_time = end_time
        self.vowel = vowel
