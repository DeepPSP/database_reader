# -*- coding: utf-8 -*-
"""
docstring, to write
"""
from typing import Tuple, List


__all__ = [
    "LCSubStr",
    "dict_depth", "dict_to_str",
]


def LCSubStr(X:str, Y:str) -> Tuple[int, List[str]]:
    """ finished, checked,

    find the longest common sub-strings of two strings,
    with complexity O(mn), m=len(X), n=len(Y)

    Parameters:
    -----------
    X, Y: str,
        the two strings to extract the longest common sub-strings

    Returns:
    --------
    lcs_len, lcs: int, list of str,
        the longest length, and the list of longest common sub-strings

    Reference:
    ----------
    https://www.geeksforgeeks.org/longest-common-substring-dp-29/
    """
    m, n = len(X), len(Y)
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]

    # To store the length of
    # longest common substring
    lcs_len = 0
    lcs = []
    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                if LCSuff[i][j] > lcs_len:
                    lcs_len = LCSuff[i][j]
                    lcs = [Y[j-lcs_len:j]]
                elif LCSuff[i][j] == lcs_len:
                    lcs_len = LCSuff[i][j]
                    lcs.append(Y[j-lcs_len:j])
            else:
                LCSuff[i][j] = 0
    return lcs_len, lcs


def dict_depth(d:dict) -> int:
    """ finished, checked,

    find the 'depth' of a (possibly) nested dict

    Parameters:
    -----------
    d: dict,
        a (possibly) nested dict
    
    Returns:
    --------
    int, the 'depth' of `d`
    """
    try:
        return 1+max([dict_depth(v) for _,v in d.items() if isinstance(v, dict)])
    except:
        return 1


def dict_to_str(d:dict, current_depth:int=1, indent_spaces:int=4) -> str:
    """ finished, checked,

    convert a (possibly) nested dict into a string of json-like format

    Parameters:
    -----------
    d: dict,
        a (possibly) nested dict
    current_depth: int, default 1,
        depth of `d` in the (possible) parent dict
    indent_spaces: int, default 0,
        the indent spaces of each depth

    Returns:
    --------
    s: str,
        the formatted string
    """
    s = "{\n"
    unit_indent = " "*indent_spaces
    prefix = unit_indent*current_depth
    for k, v in d.items():
        if isinstance(v,dict):
            s += f"{prefix}{k}: {dict_to_str(v, current_depth+1)}\n"
        else:
            s += f"{prefix}{k}: {v}\n"
    s += unit_indent*(current_depth-1)+"}"
    return s
