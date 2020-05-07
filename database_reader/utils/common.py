# -*- coding: utf-8 -*-
"""
commonly used utilities, that do not belong to a particular category
"""
import os
import subprocess
import collections
import numpy as np
import time
from logging import Logger
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
    "angle_d2r",
    "execute_cmd",
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


def angle_d2r(angle:Union[Real,np.ndarray]) -> Union[Real,np.ndarray]:
    """
    
    Parameters:
    -----------
    angle: real number or ndarray,
        the angle(s) in degrees

    Returns:
    --------
    to writereal number or ndarray, the angle(s) in radians
    """
    return np.pi*angle/180.0


def execute_cmd(cmd:str, logger:Optional[Logger]=None, raise_error:bool=True) -> Tuple[int, List[str]]:
    """
    execute shell command using `Popen`

    Parameters:
    -----------
    cmd: str,
        the shell command to be executed
    logger: Logger, optional,
    raise_error: bool, default True,
        if True, error will be raised when occured

    Returns:
    --------
    exitcode, output_msg: int, list of str,
        exitcode: exit code returned by `Popen`
        output_msg: outputs from `stdout` of `Popen`
    """
    shell_arg, executable_arg = True, None
    s = subprocess.Popen(
        cmd,
        shell=shell_arg,
        executable=executable_arg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    debug_stdout = collections.deque(maxlen=1000)
    if logger:
        logger.info("\n"+"*"*10+"  execute_cmd starts  "+"*"*10+"\n")
    while 1:
        line = s.stdout.readline().decode("utf-8", errors="replace")
        if line.rstrip():
            debug_stdout.append(line)
            if logger:
                logger.debug(line)
        exitcode = s.poll()
        if exitcode is not None:
            for line in s.stdout:
                debug_stdout.append(line.decode("utf-8", errors="replace"))
            if exitcode is not None and exitcode != 0:
                error_msg = " ".join(cmd) if not isinstance(cmd, str) else cmd
                error_msg += "\n"
                error_msg += "".join(debug_stdout)
                s.communicate()
                s.stdout.close()
                if logger:
                    logger.info("\n"+"*"*10+"  execute_cmd failed  "+"*"*10+"\n")
                if raise_error:
                    raise subprocess.CalledProcessError(exitcode, error_msg)
                else:
                    output_msg = list(debug_stdout)
                    return exitcode, output_msg
            else:
                break
    s.communicate()
    s.stdout.close()
    output_msg = list(debug_stdout)

    if logger:
        logger.info("\n"+"*"*10+"  execute_cmd succeeded  "+"*"*10+"\n")

    exitcode = 0

    return exitcode, output_msg
