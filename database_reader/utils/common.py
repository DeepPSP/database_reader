# -*- coding: utf-8 -*-
"""
commonly used utilities, that do not belong to a particular category
"""
import os
import subprocess
import collections
import numpy as np
import time
from copy import deepcopy
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
    "get_record_list_recursive",
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
    ts: int,
        timestamp, in second or millisecond, corr. to `time_string`
    """
    if return_second:
        ts = int(round(datetime.strptime(time_string, fmt).timestamp()))
    else:
        ts = int(round(datetime.strptime(time_string, fmt).timestamp()*1000))
    return ts


def modulo(val:Real, dividend:Real, val_range_start:Real=0) -> Real:
    """ finished, checked,

    find the value
        val mod dividend, within interval [val_range_start, val_range_start+abs(dividend)]

    Parameters:
    -----------
    val: real number,
        the number to be moduloed
    dividend: real number,
        the dividend
    returns:
    --------
    mod_val: real number,
        equals val mod dividend, within interval [val_range_start, val_range_start+abs(dividend)]
    """
    _dividend = abs(dividend)
    mod_val = val - val_range_start - _dividend*int((val-val_range_start)/_dividend)
    mod_val = mod_val + val_range_start if mod_val >= 0 else _dividend + mod_val + val_range_start
    return mod_val
    # alternatively
    # return (val-val_range_start)%_dividend + val_range_start


def angle_d2r(angle:Union[Real,np.ndarray]) -> Union[Real,np.ndarray]:
    """ finished, checked,
    
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
    """ finished, checked,

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


def get_record_list_recursive(db_path:str, rec_ext:str) -> List[str]:
    """ finished, checked,

    get the list of records in `db_path` recursively,
    for example, there are two folders 'patient1', 'patient2' in `db_path`,
    and there are records 'A0001', 'A0002', ... in 'patient1'; 'B0001', 'B0002', ... in 'patient2',
    then the output would be 'patient1{sep}A0001', ..., 'patient2{sep}B0001', ...,
    sep is determined by the system

    Parameters:
    -----------
    db_path: str,
        the parent (root) path of the whole database
    rec_ext: str,
        extension of the record files

    Returns:
    --------
    res: list of str,
        list of records, in lexicographical order
    """
    res = []
    db_path = os.path.join(db_path, "tmp").replace("tmp", "")  # make sure `db_path` ends with a sep
    roots = [db_path]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = [os.path.join(r, item) for item in os.listdir(r)]
            res += [item for item in tmp if os.path.isfile(item)]
            new_roots += [item for item in tmp if os.path.isdir(item)]
        roots = deepcopy(new_roots)
    res = [os.path.splitext(item)[0].replace(db_path, "") for item in res if item.endswith(rec_ext)]
    res = sorted(res)

    return res
