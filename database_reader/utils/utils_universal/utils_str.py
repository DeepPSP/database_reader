# -*- coding: utf-8 -*-
"""
docstring, to write
"""
from typing import Tuple, List


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
