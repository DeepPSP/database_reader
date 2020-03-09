# -*- coding: utf-8 -*-
"""
docstring, to write
"""

from functools import wraps


__all__ = [
    "indicator_enter_leave_func",
]


def indicator_enter_leave_func(verbose:int=0):
    """
    """
    def dec_outer(fn:callable):
        @wraps(fn)
        def dec_inner(*args, **kwargs):
            if verbose >= 1:
                print("\n"+"*"*10+"  entering function {}  ".format(fn.__name__)+"*"*10)
                start = time.time()
            response = fn(*args, **kwargs)
            if verbose >= 1:
                print("\n"+"*"*10+"  execution of function {} used {} second(s)  ".format(fn.__name__, time.time()-start)+"*"*10)
                print("\n"+"*"*10+"  leaving function {}  ".format(fn.__name__)+"*"*10+"\n")
            return response
        return dec_inner
    return dec_outer
