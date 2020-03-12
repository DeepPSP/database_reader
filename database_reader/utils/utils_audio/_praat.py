# -*- coding: utf-8 -*-
"""
improved wrapped praat Sound
"""
import parselmouth as pm
from parselmouth.praat import call
from numbers import Real
from typing import Union, Optional, List, NoReturn

from utils import ArrayLike


class PMSound(pm.Sound):
    """

    TODO: add more methods
    """
    def __init__(self, values:Optional[np.ndarray]=None, sampling_frequency:Optional[Real]=None, start_time:Optional[Real]=None, file_path:Optional[List[str]]=None):
        """
        """
        self.pm_true_false = {True: 'yes', False: 'no'}
        assert values is not None or file_path is not None
        if values is not None:
            return super().__init__(
                values=values, sampling_frequency=sampling_frequency, start_time=(start_time or 0)
            )
        else:
            return super().__init__(file_path=file_path)

    def to_formant(self, method:str='burg', time_step:Optional[real]=None, max_number_of_formants:Real=5.0, maximum_formant:Real=5500.0, window_length:Real=0.025, pre_emphasis_from:Real=50.0, number_of_std_dev:Real=1.5, maximum_number_of_iterations:Real=5, tolerance:Real=1.0e-6) -> pm.Formant:
        """
        
        Parameters:
        -----------
        method: str, default 'burg', can also be 'sl', 'keep all', 'robust',
        time_step: real number, optional, units in (s),
        max_number_of_formants: real number, default 5.0,
        maximum_formant: real number, default 5500.0, units in (Hz),
        window_length: real number, default 0.025, units in (s),
        pre_emphasis_from: real number, default 50.0, units in (Hz),
        number_of_std_dev: real number, default 1.5,
        maximum_number_of_iterations: real number, default 5,
        tolerance: real number, default 1.0e-6
        """
        m = method.lower()
        if method == 'burg':
            return self.to_formant_burg(time_step, max_number_of_formants, maximum_formant, window_length, pre_emphasis_from)
        elif method in ['sl', 'split levinson', 'split levinson (willems)']:
            return call(self, "To Formant (sl)", time_step or 0.0, max_number_of_formants, maximum_formant, window_length, pre_emphasis_from)
        elif method == 'keep all':
            return call(self, "To Formant (keep all)", time_step or 0.0, max_number_of_formants, maximum_formant, window_length, pre_emphasis_from)
        elif method == 'robust':
            return call(self, "To Formant (robust)", time_step or 0.0, max_number_of_formants, maximum_formant, window_length, pre_emphasis_from, number_of_std_dev, maximum_number_of_iterations, tolerance)
    
    def to_melspectrogram(self, window_length:Real=0.015, time_step:Real=0.005, position_of_first_filter:Real=100, distance_between_filters:Real=100, maximum_frequency:Real=0, convert_to_dB_values:bool=True) -> pm.Matrix:
        """

        Parameters:
        -----------
        window_length: real number, default 0.015, units in (s),
        time_step: real number, default 0.005, units in (s),
        position_of_first_filter: real number, default 100, units in (mel),
        distance_between_filters: real number, default 100, units in (mel),
        maximum_frequency: real number, default 0, units in (mel),
        convert_to_dB_values: bool, default True,
        """
        ms = call(self, "To MelSpectrogram", window_length, time_step, position_of_first_filter, distance_between_filters, maximum_frequency)
        ms = call(ms, "To Matrix", self.pm_true_false[convert_to_dB_values])
        return ms


    def to_power_cepstrogram(self, pitch_floor:Real=60, time_step:Real=0.002, maximum_frequency:Real=5000, pre_emphasis_from:Real=50) -> pm.Matrix:
        """

        Parameters:
        -----------
        pitch_floor: real number, default 60, units in (Hz),
        time_step: real number, default 0.002, units in (s)
        maximum_frequency: real number, default 5000, units in (Hz),
        pre_emphasis_from: real number, default 50, units in (Hz)
        """
        pc = call(self, "To PowerCepstrogram", pitch_floor, time_step, maximum_frequency, pre_emphasis_from)
        pc = call(pc, "To Matrix")
        return pc

    def to_power_cepstrum_slice(self, time:Real, pitch_floor:Real=60, time_step:Real=0.002, maximum_frequency:Real=5000, pre_emphasis_from:Real=50) -> pm.Matrix:
        """
        Parameters:
        -----------
        to write

        TODO: other operations including substract trend, smooth
        """
        pc = call(self, "To PowerCepstrogram", pitch_floor, time_step, maximum_frequency, pre_emphasis_from)
        pcs = call(pc, "To PowerCepstrum (slice)", time)
        # ref TODO before to matrix
        pcs = call(pcs, "To Matrix")

    def to_point_process_periodic_cc(self, minimum_pitch:Real=75, maximum_pitch:Real=600) -> pm.Data:
        """

        Parameters:
        -----------
        minimum_pitch: real number, default 75, units in (Hz),
        maximum_pitch: real number, default 600, units in (Hz),
        """
        return call(self, "To PointProcess (periodic, cc)", minimum_pitch, maximum_pitch)

    def to_point_process_periodic_peaks(self, minimum_pitch:Real=75, maximum_pitch:Real=600, include_maxima:bool=True, include_minima:bool=False) -> pm.Data:
        """

        Parameters:
        -----------
        minimum_pitch: real number, default 75, units in (Hz),
        maximum_pitch: real number, default 600, units in (Hz),
        include_maxima: bool, default True,
        include_minima: bool, default False,
        """
        return call(self, "To PointProcess (periodic, peaks)", minimum_pitch, maximum_pitch, self.pm_true_false[include_maxima], self.pm_true_false[include_minima])

    def to_point_process_extrema(self, channel:Union[Real, str], include_maxima:bool=True, include_minima:bool=False, interpolation:str='Sinc70') -> pm.Data:
        """

        Parameters:
        -----------
        channel: real number or str,
        include_maxima: bool, default True,
        include_minima: bool, default False,
        interpolation: str, default 'Sinc70',
        """
        assert isinstance(channel, int) or channel in ['Left', 'Right']
        assert interpolation in ['None', 'Parabolic', 'Cubic', 'Sinc70', 'Sinc700']
        return call(self, "To PointProcess (extrema)", channel, self.pm_true_false[include_maxima], self.pm_true_false[include_minima], interpolation)

    def to_point_process_zeroes(self, channel:Union[Real, str], include_raisers:bool=True, include_fallers:bool=False) -> pm.Data:
        """

        Parameters:
        -----------
        channel: real number or str,
        include_raisers: bool, default True,
        include_fallers: bool, default False,
        """
        assert isinstance(channel, int) or channel in ['Left', 'Right']
        return call(self, "To PointProcess (zeroes)", channel, self.pm_true_false[include_raisers], self.pm_true_false[include_fallers])

    def to_bark_spectrogram(self, window_length:Real=0.015, time_step:Real=0.005, position_of_first_filter:Real=1, distance_between_filters:Real=1, maximum_frequency:Real=0) -> pm.Data:
        """

        Parameters:
        -----------
        window_length: real number, default 0.015,
        time_step: real number, default 0.005,
        position_of_first_filter: real number, default 1,
        distance_between_filters: real number, default 1,
        maximum_frequency: real number, default 0,
        """
        return call(self, "To BarkSpectrogram", window_length, time_step, position_of_first_filter, distance_between_filter, maximum_frequency)

    def to_cochleagram(self, time_step:Real=0.01, frequency_resolution:Real=0.1, window_length:Real=0.003, forward_masking_time:Real=0.03) -> pm.Data:
        """

        Parameters:
        -----------
        time_step: real number, default 0.01,
        frequency_resolution: real number, default 0.1,
        window_length: real number, default 0.003,
        forward_masking_time: real number, default 0.03
        """
        return call(self, "To Cochleagram", time_step, frequency_resolution, window_length, forward_masking_time)

    def to_lpc(self, method:str, prediction_order:int=16, window_length:Real=0.025, time_step:Real=0.005, pre_emphasis_frequency:Real=50, **kwargs) -> pm.Data:
        """

        Parameters:
        -----------
        method: str,
        prediction_order: int 16,
        window_length:real number, default 0.025,
        time_step:real number, default 0.005,
        pre_emphasis_frequency:real number, default 50,
        kwargs: dict, optional,
            'tolerance1', 'tolerance2' for `method` 'marple', both default 1.0e-6
        """
        cmd = "To LPC ({})".format(method)
        if method in ['autocorrelation', 'covariance', 'burg']:
            lpc = call(self, cmd, prediction_order, window_length, time_step, pre_emphasis_frequency)
        elif method == 'marple':
            tolerance1 = kwargs.get('tolerance1', 1.0e-6)
            tolerance2 = kwargs.get('tolerance2', 1.0e-6)
            lpc = call(self, cmd, prediction_order, window_length, time_step, pre_emphasis_frequency, tolerance1, tolerance2)
        return lpc
