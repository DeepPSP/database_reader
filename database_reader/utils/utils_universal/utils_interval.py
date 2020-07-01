# -*- coding: utf-8 -*-
"""
remarks: commonly used functions related to intervals

NOTE:
    `interval` refers to interval of the form [a,b]
    `generalized_interval` refers to some (finite) union of `interval`s
TODO:
    unify `interval` and `generalized_interval`,
    by letting `interval` be of the form [[a,b]]

"""

import numpy as np
from copy import deepcopy
from functools import reduce
import time
from numbers import Real
from typing import Union, Optional, Any, List, Tuple

from ..common import ArrayLike


__all__ = [
    "get_optimal_covering",
    "overlaps",
    "validate_interval",
    "in_interval",
    "in_generalized_interval",
    "get_confidence_interval",
    "intervals_union",
    "generalized_intervals_union",
    "intervals_intersection",
    "generalized_intervals_intersection",
    "generalized_interval_complement",
    "find_max_cont_len",
    "interval_len",
    "generalized_interval_len",
    "diff_with_step",
    "find_extrema",
    "is_intersect",
]


EMPTY_SET = []
Interval = Union[List[Real], Tuple[Real], type(EMPTY_SET)]
GeneralizedInterval = Union[List[Interval], Tuple[Interval], type(EMPTY_SET)]


def overlaps(interval:Interval, another:Interval) -> int:
    """ finished, checked,

    Return the amount of overlap, in bp between interval and anohter.
    If >0, the number of bp of overlap
    If 0,  they are book-ended
    If <0, the distance in bp between them

    Parameters:
    -----------
    interval, another: two `Interval`s

    Returns:
    --------
    int, overlap length of two intervals; if < 0, the distance of two intervals
    """
    # in case a or b is not in ascending order
    interval.sort()
    another.sort()
    return min(interval[-1], another[-1]) - max(interval[0], another[0])


def validate_interval(interval:Union[Interval, GeneralizedInterval], join_book_endeds:bool=True) -> Tuple[bool,Union[Interval, GeneralizedInterval]]:
    """ finished, not checked,

    check whether `interval` is an `Interval` or a `GeneralizedInterval`,
    if true, return True, and validated (of the form [a,b] with a<=b) interval,
    return `False, []`, otherwise

    Parameters:
    -----------
    interval: Interval, or unions of `Interval`s
    join_book_endeds: bool, default True,
        if True, two book-ended intervals will be joined into one

    Returns:
    --------
    tuple, consisting of
        a bool, indicating whether `interval` is a valid interval
        an interval (can be empty)
    """
    if isinstance(interval[0], (list,tuple)):
        info = [validate_interval(itv,join_book_endeds) for itv in interval]
        if all([item[0] for item in info]):
            return True, intervals_union(interval,join_book_endeds)
        else:
            return False, []

    if len(interval) == 2:
        return True, [min(interval), max(interval)]
    else:
        return False, []


def in_interval(val:Real, interval:Interval) -> bool:
    """ finished, checked,

    check whether val is inside interval or not

    Parameters:
    -----------
    val: real number,
    interval: Interval,

    Returns:
    --------
    bool,
    """
    interval.sort()
    return True if interval[0] <= val <= interval[-1] else False


def in_generalized_interval(val:Real, generalized_interval:GeneralizedInterval) -> bool:
    """ finished, checked,

    check whether val is inside generalized_interval or not

    Parameters:
    -----------
    val: real number,
    generalized_interval: union of `Interval`s,

    Returns:
    --------
    bool,
    """
    for interval in generalized_interval:
        if in_interval(val, interval):
            return True
    return False


def get_confidence_interval(data:Optional[ArrayLike]=None, val:Optional[Real]=None, rmse:Optional[float]=None, confidence:float=0.95, **kwargs) -> np.ndarray:
    """ finished, checked,

    Parameters:
    -----------
    data: array_like, optional,
    val: real number, optional,
    rmse: float, optional,
    confidence: float, default 0.95,
    kwargs: dict,

    Returns:
    --------
    conf_itv: ndarray,
    """
    from scipy.stats import norm
    assert data or (val and rmse), "insufficient data for computing"
    correct_factor = kwargs.get('correct_factor', 1)
    bias = norm.ppf(0.5 + confidence / 2)
    if data is None:
        lower_bound = (val - rmse * bias) * correct_factor
        upper_bound = (val + rmse * bias) / correct_factor
    else:
        average = np.mean(np.array(data))
        std = np.std(np.array(data), ddof=1)
        lower_bound = (average - std * bias) * correct_factor
        upper_bound = (average + std * bias) / correct_factor
    conf_itv = np.array([lower_bound, upper_bound])
    return conf_itv


def intervals_union(interval_list:GeneralizedInterval, join_book_endeds:bool=True) -> GeneralizedInterval:
    """ finished, checked,

    find the union (ordered and non-intersecting) of all the intervals in `interval_list`,
    which is a list of intervals in the form [a,b], where a,b need not be ordered

    Parameters:
    -----------
    interval_list: GeneralizedInterval,
        the list of intervals to calculate their union
    join_book_endeds: bool, default True,
        join the book-ended intervals into one (e.g. [[1,2],[2,3]] into [1,3]) or not
    
    Returns:
    --------
    GeneralizedInterval, the union of the intervals in `interval_list`
    """
    interval_sort_key = lambda i: i[0]
    # list_add = lambda list1, list2: list1+list2
    processed = [item for item in interval_list if len(item) > 0]
    for item in processed:
        item.sort()
    processed.sort(key=interval_sort_key)
    # end_points = reduce(list_add, processed)
    merge_flag = True
    while merge_flag:
        merge_flag = False
        new_intervals = []
        if len(processed) == 1:
            return processed
        for idx, interval in enumerate(processed[:-1]):
            this_start, this_end = interval
            next_start, next_end = processed[idx + 1]
            # it is certain that this_start <= next_start
            if this_end < next_start:
                # 两区间首尾分开
                new_intervals.append([this_start, this_end])
                if idx == len(processed) - 2:
                    new_intervals.append([next_start, next_end])
            elif this_end == next_start:
                # 两区间首尾正好在一点
                # 需要区别对待单点区间以及有长度的区间
                # 以及join_book_endeds的情况
                # 来判断是否合并
                if (this_start == this_end or next_start == next_end) or join_book_endeds:
                    # 单点区间以及join_book_endeds为True时合并
                    new_intervals.append([this_start, max(this_end, next_end)])
                    new_intervals += processed[idx + 2:]
                    merge_flag = True
                    processed = new_intervals
                    break
                else:
                    # 都是有长度的区间且join_book_endeds为False则不合并
                    new_intervals.append([this_start, this_end])
                    if idx == len(processed) - 2:
                        new_intervals.append([next_start, next_end])
            else:
                new_intervals.append([this_start, max(this_end, next_end)])
                new_intervals += processed[idx + 2:]
                merge_flag = True
                processed = new_intervals
                break
        processed = new_intervals
    return processed


def generalized_intervals_union(interval_list:Union[List[GeneralizedInterval],Tuple[GeneralizedInterval]], join_book_endeds:bool=True) -> GeneralizedInterval:
    """ finished, checked,

    calculate the union of a list (or tuple) of `GeneralizedInterval`s

    Parameters:
    -----------
    interval_list: list or tuple,
        a list (or tuple) of `GeneralizedInterval`s
    join_book_endeds: bool, default True,
        join the book-ended intervals into one (e.g. [[1,2],[2,3]] into [1,3]) or not

    Returns:
    --------
    GeneralizedInterval, the union of `interval_list`
    """
    all_intervals = [itv for gnr_itv in interval_list for itv in gnr_itv]
    return intervals_union(interval_list=all_intervals, join_book_endeds=join_book_endeds)


def intervals_intersection(interval_list:GeneralizedInterval, drop_degenerate:bool=True) -> Interval:
    """ finished, checked,

    calculate the intersection of all intervals in interval_list

    Parameters:
    -----------
    interval_list: GeneralizedInterval,
        the list of intervals to yield intersection
    drop_degenerate: bool, default True,
        whether or not drop the degenerate intervals, i.e. intervals with length 0
    
    Returns:
    --------
    Interval, the intersection of all intervals in `interval_list`
    """
    if [] in interval_list:
        return []
    for item in interval_list:
        item.sort()
    potential_start = max([item[0] for item in interval_list])
    potential_end = min([item[-1] for item in interval_list])
    if (potential_end > potential_start) or (potential_end == potential_start and not drop_degenerate):
        return [potential_start, potential_end]
    else:
        return []


def generalized_intervals_intersection(generalized_interval:GeneralizedInterval, another_generalized_interval:GeneralizedInterval, drop_degenerate:bool=True) -> GeneralizedInterval:
    """ finished, checked,

    calculate the intersection of generalized_interval and another_generalized_interval,
    which are both generalized intervals

    Parameters:
    -----------
    generalized_interval, another_generalized_interval: GeneralizedInterval,
        the 2 `GeneralizedInterval`s to yield intersection
    drop_degenerate: bool, default True,
        whether or not drop the degenerate intervals, i.e. intervals with length 0
    
    Returns:
    --------
    a GeneralizedInterval, the intersection of `generalized_interval` and `another_generalized_interval`
    """
    this = intervals_union(generalized_interval)
    another = intervals_union(another_generalized_interval)
    # 注意，此时this, another都是按区间起始升序排列的，
    # 而且这二者都是一系列区间的不交并
    ret = []
    # 以下流程可以优化
    cut_idx = 0
    for item in this:
        another = another[cut_idx:]
        intersected_indices = []
        for idx, item_prime in enumerate(another):
            tmp = intervals_intersection([item,item_prime], drop_degenerate=drop_degenerate)
            if len(tmp) > 0:
                ret.append(tmp)
                intersected_indices.append(idx)
        if len(intersected_indices) > 0:
            cut_idx = intersected_indices[-1]
    return ret


def generalized_interval_complement(total_interval:Interval, generalized_interval:GeneralizedInterval) -> GeneralizedInterval:
    """ finished, checked, to be improved,

    TODO: the case `total_interval` is a `GeneralizedInterval`

    Parameters:
    -----------
    total_interval, Interval,
    generalized_interval: union of `Interval`s

    Returns:
    --------
    cpl: union of `Interval`s,
        the complement of `generalized_interval` in `total_interval`
    """
    rearranged_intervals = intervals_union(generalized_interval)
    total_interval.sort()
    tot_start, tot_end = total_interval[0], total_interval[-1]
    rearranged_intervals = [
        [max(tot_start, item[0]), min(tot_end, item[-1])] \
            for item in rearranged_intervals if overlaps(item, total_interval) > 0
    ]
    slice_points = [tot_start]
    for item in rearranged_intervals:
        slice_points += item
    slice_points.append(tot_end)
    cpl = []
    for i in range(len(slice_points) // 2):
        if slice_points[2 * i + 1] - slice_points[2 * i] > 0:
            cpl.append([slice_points[2 * i], slice_points[2 * i + 1]])
    return cpl


def get_optimal_covering(total_interval:Interval, to_cover:list, min_len:int, split_threshold:int, traceback:bool=False, **kwargs) -> Tuple[GeneralizedInterval,list]:
    """ finished, checked,

    获取覆盖to_cover中每一项的满足min_len, split_threshold条件的最佳覆盖

    Parameters:
    -----------
    total_interval: 总的大区间
    to_cover: 需要覆盖的点和区间的列表
    min_len: 每一个覆盖的最小长度
    split_threshold: 覆盖之间的最小距离
    traceback: 是否记录每个covering覆盖了的to_cover的项（的index）
    注意单位保持一致！
    如果to_cover的范围超过total_interval的范围，会抛出异常

    Returns:
    --------
    (ret, ret_traceback)
        ret是一个GeneralizedInterval，满足min_len, split_threshold的条件；
        ret_traceback是一个list，
        其中每一项是一个list，记录了ret中对应的interval覆盖的to_cover中的项的indices
    """
    start_time = time.time()
    verbose = kwargs.get('verbose', 0)
    tmp = sorted(total_interval)
    tot_start, tot_end = tmp[0], tmp[-1]

    if verbose >= 1:
        print('total_interval =', total_interval, 'with_length =', tot_end-tot_start)

    if tot_end - tot_start < min_len:
        ret = [[tot_start, tot_end]]
        ret_traceback = [list(range(len(to_cover)))] if traceback else []
        return ret, ret_traceback
    to_cover_intervals = []
    for item in to_cover:
        if isinstance(item, list):
            to_cover_intervals.append(item)
        else:
            to_cover_intervals.append([item, item])
    if traceback:
        replica_for_traceback = deepcopy(to_cover_intervals)

    if verbose >= 2:
        print('to_cover_intervals after all converted to intervals', to_cover_intervals)

        # elif isinstance(item, int):
        #     to_cover_intervals.append([item, item+1])
        # else:
        #     raise ValueError("{0} is not an integer or an interval".format(item))
    # to_cover_intervals = interval_union(to_cover_intervals)

    for interval in to_cover_intervals:
        interval.sort()
    
    interval_sort_key = lambda i: i[0]
    to_cover_intervals.sort(key=interval_sort_key)

    if verbose >= 2:
        print('to_cover_intervals after sorted', to_cover_intervals)

    # if to_cover_intervals[0][0] < tot_start or to_cover_intervals[-1][-1] > tot_end:
    #     raise IndexError("some item in to_cover list exceeds the range of total_interval")
    # these cases now seen normal, and treated as follows:
    for item in to_cover_intervals:
        item[0] = max(item[0], tot_start)
        item[-1] = min(item[-1], tot_end)
    # to_cover_intervals = [item for item in to_cover_intervals if item[-1] > item[0]]

    # 确保第一个区间的末尾到tot_start的距离不低于min_len
    to_cover_intervals[0][-1] = max(to_cover_intervals[0][-1], tot_start + min_len)
    # 确保最后一个区间的起始到tot_end的距离不低于min_len
    to_cover_intervals[-1][0] = min(to_cover_intervals[-1][0], tot_end - min_len)

    if verbose >= 2:
        print('to_cover_intervals after two tails adjusted', to_cover_intervals)

    # 将间隔（有可能是负的，即有重叠）小于split_threshold的区间合并
    merge_flag = True
    while merge_flag:
        merge_flag = False
        new_intervals = []
        if len(to_cover_intervals) == 1:
            break
        for idx, item in enumerate(to_cover_intervals[:-1]):
            this_start, this_end = item
            next_start, next_end = to_cover_intervals[idx + 1]
            if next_start - this_end >= split_threshold:
                if split_threshold == (next_start - next_end) == 0 or split_threshold == (this_start - this_end) == 0:
                    # 需要单独处理 split_threshold ==0 以及正好有连着的单点集这种情况
                    new_intervals.append([this_start, max(this_end, next_end)])
                    new_intervals += to_cover_intervals[idx + 2:]
                    merge_flag = True
                    to_cover_intervals = new_intervals
                    break
                else:
                    new_intervals.append([this_start, this_end])
                    if idx == len(to_cover_intervals) - 2:
                        new_intervals.append([next_start, next_end])
            else:
                new_intervals.append([this_start, max(this_end, next_end)])
                new_intervals += to_cover_intervals[idx + 2:]
                merge_flag = True
                to_cover_intervals = new_intervals
                break
    if verbose >= 2:
        print('to_cover_intervals after merging intervals whose gaps < split_threshold', to_cover_intervals)

    # 此时，to_cover_intervals中所有区间的间隔都大于split_threshold
    # 但是除了头尾两个区间之外的区间的长度可能小于min_len
    ret = []
    ret_traceback = []
    if len(to_cover_intervals) == 1:
        # 注意，此时to_cover_intervals只有一个，这个元素（区间）的长度应该不小于min_len
        # 保险起见还是计算一下
        mid_pt = (to_cover_intervals[0][0]+to_cover_intervals[0][-1]) // 2
        half_len = min_len // 2
        if mid_pt - tot_start < half_len:
            ret_start = tot_start
            ret_end = min(tot_end, max(tot_start+min_len, to_cover_intervals[0][-1]))
            ret = [[ret_start, ret_end]]
        else:
            ret_start = max(tot_start, min(to_cover_intervals[0][0], mid_pt-half_len))
            ret_end = min(tot_end, max(mid_pt-half_len+min_len, to_cover_intervals[0][-1]))
            ret = [[ret_start, ret_end]]

    start = min(to_cover_intervals[0][0], to_cover_intervals[0][-1]-min_len)

    for idx, item in enumerate(to_cover_intervals[:-1]):
        # print('item', item)
        this_start, this_end = item
        next_start, next_end = to_cover_intervals[idx + 1]
        potential_end = max(this_end, start + min_len)
        # print('start', start)
        # print('potential_end', potential_end)
        # 如果potential_end到next_start的间隔不够长，
        # 则进入下一循环（如果不到to_cover_intervals尾部）
        if next_start - potential_end < split_threshold:
            if idx < len(to_cover_intervals) - 2:
                continue
            else:
                # 此时 idx==len(to_cover_intervals)-2
                # next_start (从而start也是) 到 tot_end 距离至少为min_len
                ret.append([start, max(start + min_len, next_end)])
        else:
            ret.append([start, potential_end])
            start = next_start
            if idx == len(to_cover_intervals) - 2:
                ret.append([next_start, max(next_start + min_len, next_end)])
        # print('ret', ret)
    if traceback:
        for item in ret:
            record = []
            for idx, item_prime in enumerate(replica_for_traceback):
                itc = intervals_intersection([item, item_prime])
                len_itc = itc[-1] - itc[0] if len(itc) > 0 else -1
                if len_itc > 0 or (len_itc == 0 and item_prime[-1] - item_prime[0] == 0):
                    record.append(idx)
            ret_traceback.append(record)
    
    if verbose >= 1:
        print('the final result of get_optimal_covering is ret = {0}, ret_traceback = {1}, the whole process used {2} second(s)'.format(ret, ret_traceback, time.time()-start_time))
    
    return ret, ret_traceback


def find_max_cont_len(sub_list:Interval, tot_len:Real) -> dict:
    """ finished, checked,

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write

    设sub_list为[0,1,2,...,tot_len-1]的一个子列，
    计算sub_list的最常的连续子列的长度，该子列在sub_list中起始位置，以及该最长连续子列
    例如，tot_len=10, sub_list=[0,2,3,4,7,9],
    那么返回3, 1, [2,3,4]
    """
    complementary_sub_list = [-1] + [i for i in range(tot_len) if i not in sub_list] + [tot_len]
    diff_list = np.diff(np.array(complementary_sub_list))
    max_cont_len = np.max(diff_list) - 1
    max_cont_sub_list_start = np.argmax(diff_list)
    max_cont_sub_list = sub_list[max_cont_sub_list_start: max_cont_sub_list_start + max_cont_len]
    ret = {
        'max_cont_len': max_cont_len,
        'max_cont_sub_list_start': max_cont_sub_list_start,
        'max_cont_sub_list': max_cont_sub_list
    }
    return ret


def interval_len(interval:Interval) -> Real:
    """ finished, checked,

    compute the length of an interval. -1 for the empty interval []

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write
    """
    interval.sort()
    return interval[-1] - interval[0] if len(interval) > 0 else -1


def generalized_interval_len(generalized_interval:GeneralizedInterval) -> Real:
    """ finished, checked,

    compute the length of a generalized interval. -1 for the empty interval []
    """
    return sum([interval_len(item) for item in intervals_union(generalized_interval)])


def diff_with_step(a:ArrayLike, step:int=1, **kwargs) -> np.ndarray:
    """ finished, checked,

    compute a[n+step] - a[n] for all valid n

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write
    """
    return np.array([a[n+step]-a[n] for n in range(len(a)-step)])


def find_extrema(signal:Optional[ArrayLike]=None, mode:str='both') -> np.ndarray:
    """
    Locate local extrema points in a signal. Based on Fermat's Theorem

    Parameters:
    -----------
    signal : array
        Input signal.
    mode : str, optional
        Whether to find maxima ('max'), minima ('min'), or both ('both').
    
    Returns:
    --------
    extrema : array
        Indices of the extrama points.
    values : array
        Signal values at the extrema points.
    """
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if mode not in ['max', 'min', 'both']:
        raise ValueError("Unknwon mode %r." % mode)

    aux = np.diff(np.sign(np.diff(signal)))

    if mode == 'both':
        aux = np.abs(aux)
        extrema = np.nonzero(aux > 0)[0] + 1
    elif mode == 'max':
        extrema = np.nonzero(aux < 0)[0] + 1
    elif mode == 'min':
        extrema = np.nonzero(aux > 0)[0] + 1
        
    return extrema


def is_intersect(interval:Union[GeneralizedInterval,Interval], another_interval:Union[GeneralizedInterval,Interval]) -> bool:
    """

    determines if two (generalized) intervals intersect or not

    Parameters:
    -----------
    interval, another_interval: GeneralizedInterval or Interval

    Returns:
    --------
    bool, True if `interval` intersects with another_interval, False otherwise
    """
    if interval is None or another_interval is None or len(interval)*len(another_interval)==0:
        # the case of empty set
        return False
        
    # check if is GeneralizedInterval
    is_generalized = isinstance(interval[0], (list,tuple))
    is_another_generalized = isinstance(another_interval[0], (list,tuple))

    if is_generalized and is_another_generalized:
        return any([is_intersect(interval, itv) for itv in another_interval])
    elif not is_generalized and is_another_generalized:
        return is_intersect(another_interval, interval)
    elif is_generalized:  # and not is_another_generalized
        return any([is_intersect(itv, another_interval) for itv in interval])
    else:  # not is_generalized and not is_another_generalized
        return any([overlaps(interval, another_interval)>0])
