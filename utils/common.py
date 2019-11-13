"""
commonly used utilities, that do not belong to a particular category
"""

import numpy as np
import time
from datetime import datetime, timedelta
from typing import Union, Optional, Any, Iterable, List, Tuple, Dict, Callable, NoReturn
from numbers import Real


__all__ = [
    "ArrayLike", "ArrayLike_Float", "ArrayLike_Int",
    "MilliSecond", "Second",
    "DEFAULT_FIG_SIZE_PER_SEC",
    "idx_to_ts",
    "timestamp_to_local_datetime_string",
    "modulo",
]


ArrayLike = Union[list,tuple,np.ndarray]
ArrayLike_Float = Union[List[float],Tuple[float],np.ndarray]
ArrayLike_Int = Union[List[int],Tuple[int],np.ndarray]
MilliSecond = int
Second = int


DEFAULT_FIG_SIZE_PER_SEC = 4.8


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


def modulo(val:Real, dividend:Real, val_range_start:Real=0) -> Real:
    """
    returns:
        val mod dividend, positive,
        and within interval [val_range_start, val_range_start+abs(dividend)]
    """
    _dividend = abs(dividend)
    ret = val-val_range_start-_dividend*int((val-val_range_start)/_dividend)
    return ret+val_range_start if ret >= 0 else _dividend+ret+val_range_start
    # alternatively
    # return (val-val_range_start)%_dividend + val_range_start


def filter_by_percentile(s:ArrayLike, q:Union[int,List[int]], return_mask:bool=False) -> Union[np.ndarray,Tuple[np.ndarray,np.ndarray]]:
    """

    Parameters:
    -----------
    to write
    """
    _s = np.array(s)
    original_shape = _s.shape
    _s = _s.reshape(-1, _s.shape[-1])  # flatten, but keep the last dim
    l,d = _s.shape
    _q = sorted(q) if isinstance(q,list) else [(100-q)//2, (100+q)//2]
    iqrs = np.percentile(_s, _q, axis=0)
    validity = np.full(shape=l, fill_value=True, dtype=bool)
    for idx in range(d):
        validity = (validity) & (_s[...,idx] >= iqrs[...,idx][0]) & (_s[...,idx] <= iqrs[...,idx][-1])
    if return_mask:
        return _s[validity], validity.reshape(original_shape[:-1])
    else:
        return _s[validity]
