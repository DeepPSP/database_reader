# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import numpy as np
import time
from datetime import datetime, timedelta
from typing import Union, Optional, Any, Iterable, List, Tuple, Dict, Callable, NoReturn
from numbers import Real


__all__ = [
    "ArrayLike", "ArrayLike_Float", "ArrayLike_Int",
    "MilliSecond", "Second",
    "idx_to_ts",
    "timestamp_to_local_datetime_string",
]


ArrayLike = Union[list,tuple,np.ndarray]
ArrayLike_Float = Union[List[float],Tuple[float],np.ndarray]
ArrayLike_Int = Union[List[int],Tuple[int],np.ndarray]
MilliSecond = int
Second = int


def idx_to_ts(idx:int, start_ts:MilliSecond, fs:int) -> MilliSecond:
    """ finished, checked,
    
    Parameters:
    -----------
    idx, int,
        the index to be converted into timestamp
    start_ts, int,
        the timestamp of the point at index 0
    fs: int,
        sampling frequency

    Returns:
    --------
    int, the timestamp of the point at index `idx`
    """
    return int(start_ts + idx * 1000 // fs)


def timestamp_to_local_datetime_string(ts:int, ts_in_second:bool=False, fmt:str="%Y-%m-%d %H:%M:%S") -> str:
    """ finished, checked,

    Parameters:
    -----------
    ts: int,
        timestamp, in second or millisecond
    ts_in_second, bool, default False,
        if Ture, `ts` is in second, otherwise in millisecond
    fmt, str, default "%Y-%m-%d %H:%M:%S",
        the format of the output string

    Returns:
    --------
    str, the string form of `ts` in the form of `fmt`
    """
    from dateutil import tz

    if ts_in_second:
        utc = datetime.utcfromtimestamp(ts)
    else:
        utc = datetime.utcfromtimestamp(ts // 1000)
    
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    # Tell the datetime object that it's in UTC time zone since 
    # datetime objects are 'naive' by default
    utc = utc.replace(tzinfo=from_zone)

    # Convert time zone
    return utc.astimezone(to_zone).strftime(fmt)


def time_string_to_timestamp(time_string:str, fmt:str="%Y-%m-%d %H:%M:%S", return_second:bool=False) -> int:
    """ finished, checked,

    Parameters:
    -----------
    time_string: str,
        the time in the string format to be converted
    fmt: str, default "%Y-%m-%d %H:%M:%S",
        the format of `time_string`
    return_second: bool, default False,
        if True, the output is in second, otherwise in millisecond

    Returns:
    --------
    int, timestamp, in second or millisecond, corr. to `time_string`
    """
    if return_second:
        return int(round(datetime.strptime(time_string, fmt).timestamp()))
    else:
        return int(round(datetime.strptime(time_string, fmt).timestamp()*1000))
