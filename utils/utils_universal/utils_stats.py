"""
frequently used functions for statistics, mainly which have been implemented in matlab and R, rather than in python
"""
import numpy as np
import pandas as pd
import random
from math import sqrt, atan2, factorial
# from numpy import pi as PI
from scipy import stats as ss
from scipy import linalg
from scipy.linalg import solve_triangular
from scipy.spatial import distance as scipy_dist
import warnings

from numbers import Real
from typing import Union, Optional, List, Tuple
from ..common import ArrayLike


__all__ = [
    "gamrnd",
    "autocorr",
    "kmeans2_is_correct",
    "is_outlier",
    "_support_singular",
    "log_multivariate_normal_density",
    "mahalanobis",
    "log_likelihood",
    "likelihood",
    "covariance_ellipse",
    "_eigsorted",
    "rand_student_t",
    "samp_ent",
    "shannon_entropy",
    "sample_entropy",
    "multiscale_entropy",
    "permutation_entropy",
    "multiscale_permutation_entropy",
    "filter_by_percentile",
    "train_test_split_dataframe",
]


# Older versions of scipy do not support the allow_singular keyword. I could
# check the version number explicily, but perhaps this is clearer
_support_singular = True
try:
    ss.multivariate_normal.logpdf(1, 1, 1, allow_singular=True)
except TypeError:
    warnings.warn(
        'You are using a version of SciPy that does not support the '\
        'allow_singular parameter in scipy.stats.multivariate_normal.logpdf(). '\
        'Future versions of FilterPy will require a version of SciPy that '\
        'implements this keyword',
        DeprecationWarning)
    _support_singular = False


def autocorr(x:ArrayLike, normalize:bool=False) -> np.ndarray:
    """ 已完成，已检查

    autocorrelation of the time series x
    """
    if normalize:
        _x = np.array(x) - np.mean(x)
        result = np.correlate(_x, _x, mode='full')[result.size//2:]
        result = result / np.sum(np.power(_x, 2))
    else:
        result = np.correlate(x, x, mode='full')[result.size//2:]
    return result


def kmeans2_is_correct(data:np.ndarray, centroids:np.ndarray, labels:np.ndarray, verbose:int=0) -> bool:
    """ 已完成，已检查

    检查from scipy.cluster.vq.kmeans2的结果是否正确
    """
    nb_clusters = len(centroids)
    nb_clusters2 = len(set(labels))

    if verbose >= 1:
        print('nb_clusters(len(centroids)) = {0}, nb_clusters2(len(set(labels))) = {1}'.format(nb_clusters,nb_clusters2))
        if verbose >= 2:
            print('data =',data)
            print('centroids =', centroids)
            print('labels =', labels)
    
    if nb_clusters != nb_clusters2:
        return False
    
    if nb_clusters == 1:
        if np.nanmax(data)/np.nanmin(data)>=1.5:
            return False
        else:
            return True
    
    to_check = [lb for lb in range(nb_clusters) if (labels==lb).sum()>1]
    if verbose >= 1:
        print('to_check =', to_check)
        print('np.sign(data-centroids[lb]) =', [np.sign(data-centroids[lb]) for lb in to_check])
    return all([len(set(np.sign(data-centroids[lb])))>=2 for lb in to_check])  # == 2 or == 3


def is_outlier(to_check_val:Real, normal_vals:Union[List[int],List[float],Tuple[int],Tuple[float],np.ndarray], verbose:int=0) -> bool:
    """ 已完成，已检查

    check if to_check_val is an outlier in normal_vals
    """
    perc75, perc25 = np.percentile(normal_vals, [75,25])
    iqr = perc75 - perc25
    lower_bound = perc25 - 1.5 * iqr
    upper_bound = perc75 + 1.5 * iqr
    if verbose >= 1:
        print('75 percentile = {0}, 25 percentile = {1}, iqr = {2}, lower_bound = {3}, upper_bound = {4}'.format(perc75, perc25, iqr, lower_bound, upper_bound))
    return not lower_bound <= to_check_val <= upper_bound


def log_multivariate_normal_density(X:ArrayLike, means:ArrayLike, covars:ArrayLike, min_covar:float=1.e-7) -> np.ndarray:
    """
    
    Log probability for full covariance matrices.
    """
    _X =  np.array(X)
    n_samples, n_dim = _X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv)
        except linalg.LinAlgError:
            # The model is most probabily stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim))
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = solve_triangular(cv_chol, (_X - mu).T).T
        log_prob[:, c] = - 0.5 * (np.sum(cv_sol ** 2, axis=1) + n_dim*np.log(2*np.pi) + cv_log_det)

    return log_prob


def mahalanobis(x:Union[list,tuple,np.ndarray,float], mean:Union[list,tuple,np.ndarray,float], cov:Union[list,tuple,np.ndarray,float]) -> float:
    """
    Computes the Mahalanobis distance between the state vector x from the
    Gaussian `mean` with covariance `cov`. This can be thought as the number
    of standard deviations x is from the mean, i.e. a return value of 3 means
    x is 3 std from mean.
    Parameters
    ----------
    x : (N,) array_like, or float
        Input state vector
    mean : (N,) array_like, or float
        mean of multivariate Gaussian
    cov : (N, N) array_like  or float
        covariance of the multivariate Gaussian
    Returns
    -------
    mahalanobis : double
        The Mahalanobis distance between vectors `x` and `mean`
    Examples
    --------
    >>> mahalanobis(x=3., mean=3.5, cov=4.**2) # univariate case
    0.125
    >>> mahalanobis(x=3., mean=6, cov=1) # univariate, 3 std away
    3.0
    >>> mahalanobis([1., 2], [1.1, 3.5], [[1., .1],[.1, 13]])
    0.42533327058913922
    """

    _x = scipy_dist._validate_vector(x)
    _mean = scipy_dist._validate_vector(mean)

    if _x.shape != _mean.shape:
        raise ValueError("length of input vectors must be the same")

    y = _x - _mean
    S = np.atleast_2d(cov)

    dist = float(np.dot(np.dot(y.T, np.linalg.inv(S)), y))
    return sqrt(dist)


def log_likelihood(z:Union[np.ndarray,float,int], x:np.ndarray, P:np.ndarray, H:np.ndarray, R:np.ndarray) -> Union[np.ndarray,float,int]:
    """
    Returns log-likelihood of the measurement z given the Gaussian
    posterior (x, P) using measurement function H and measurement
    covariance error R
    """
    S = np.dot(H, np.dot(P, H.T)) + R
    return ss.multivariate_normal.logpdf(z, np.dot(H, x), S)


def likelihood(z:Union[np.ndarray,float,int], x:np.ndarray, P:np.ndarray, H:np.ndarray, R:np.ndarray) -> Union[np.ndarray,float,int]:
    """
    Returns likelihood of the measurement z given the Gaussian
    posterior (x, P) using measurement function H and measurement
    covariance error R
    """
    return np.exp(log_likelihood(z, x, P, H, R))


def covariance_ellipse(P, deviations=1):
    """
    Returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.
    Parameters
    ----------
    P : nd.array shape (2,2)
       covariance matrix
    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.
    Returns (angle_radians, width_radius, height_radius)
    """

    U, s, _ = linalg.svd(P)
    orientation = atan2(U[1, 0], U[0, 0])
    width = deviations * sqrt(s[0])
    height = deviations * sqrt(s[1])

    if height > width:
        raise ValueError('width must be greater than height')

    return (orientation, width, height)


def _eigsorted(cov, asc=True):
    """
    Computes eigenvalues and eigenvectors of a covariance matrix and returns
    them sorted by eigenvalue.
    Parameters
    ----------
    cov : ndarray
        covariance matrix
    asc : bool, default=True
        determines whether we are sorted smallest to largest (asc=True),
        or largest to smallest (asc=False)
    Returns
    -------
    eigval : 1D ndarray
        eigenvalues of covariance ordered largest to smallest
    eigvec : 2D ndarray
        eigenvectors of covariance matrix ordered to match `eigval` ordering.
        I.e eigvec[:, 0] is the rotation vector for eigval[0]
    """

    eigval, eigvec = np.linalg.eigh(cov)
    order = eigval.argsort()
    if not asc:
        # sort largest to smallest
        order = order[::-1]

    return eigval[order], eigvec[:, order]


def rand_student_t(df:Real, mu:Real=0, std:Real=1) -> Real:
    """
    return random number distributed by student's t distribution with
    `df` degrees of freedom with the specified mean and standard deviation.
    """

    x = random.gauss(0, std)
    y = 2.0*random.gammavariate(0.5 * df, 2.0)
    return x / (sqrt(y / df)) + mu


def samp_ent(s:ArrayLike, m:int, r:Real) -> float:
    """ 已完成，

    sample entropy of signal `s`, equals ent.sample_entropy(s, m, r)[-1], a simple version

    Parameters:
    -----------
    s: array_like,
        the signal (time series)
    m: int,
        the sample length
    r: real number,
        tolerance

    Returns:
    --------
    sample entropy, ratio "#templates of length k+1" / "#templates of length k"

    copied from
        https://en.wikipedia.org/wiki/Sample_entropy#Implementation
    """
    def _maxdist(x_i, x_j):
        result = max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        return result

    def _phi(m):
        N = len(s)
        x = [[s[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = 0
        for i in range(len(x)):
            for j in range(len(x)):
                if i == j:
                    continue
                C += (_maxdist(x[i], x[j]) <= r)
        return C

    # N = len(s)
    
    return -np.log(_phi(m+1) / _phi(m))


def shannon_entropy(s:ArrayLike) -> float:
    """Return the Shannon Entropy of the sample data.

    Args:
        s: Vector or string of the sample data

    Returns:
        The Shannon Entropy as float value
    """
    # Create a frequency data
    data_set = list(set(s))
    freq_list = []
    for entry in data_set:
        counter = 0
        for i in s:
            if i == entry:
                counter += 1
        freq_list.append(counter / len(s))

    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    ent = -ent

    return ent


def sample_entropy(s:ArrayLike, sample_length:int, tolerance:Optional[Real]=None) -> np.ndarray:
    """
    
    calculates the sample entropy of degree m (m+1=sample_length) of the time series `s`

    This method uses chebychev norm.
    It is quite fast for random data, but can be slower is there is
    structure in the input time series.

    Parameters:
    -----------
    s: array_like
        the time series for analysis
    sample_length: int,
        length of longest template vector
    tolerance: real number, optional,
        tolerance (defaults to 0.1 * std(time_series)))
    
    Returns:
    --------
    np.ndarray, the vector of sample entropies

        SE[k] is ratio "#templates of length k+1" / "#templates of length k"
        where #templates of length 0" = n*(n - 1) / 2, by definition
    
    Note:
    -----
        The parameter 'sample_length' is equal to m + 1 in Ref[1].

    References:
    -----------
    [1] http://en.wikipedia.org/wiki/Sample_Entropy
    [2] http://physionet.incor.usp.br/physiotools/sampen/
    [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis of biological signals
    """
    #The code below follows the sample length convention of Ref [1] so:
    M = sample_length - 1

    time_series = np.array(s)
    if tolerance is None:
        tolerance = 0.1*np.std(time_series)

    n = len(time_series)

    #Ntemp is a vector that holds the number of matches. N[k] holds matches templates of length k
    Ntemp = np.zeros(M + 2)
    #Templates of length 0 matches by definition:
    Ntemp[0] = n*(n - 1) / 2


    for i in range(n - M - 1):
        template = time_series[i:(i+M+1)]  # We have 'M+1' elements in the template
        rem_time_series = time_series[i+1:]

        searchlist = np.nonzero(np.abs(rem_time_series - template[0]) < tolerance)[0]

        go = len(searchlist) > 0

        length = 1

        Ntemp[length] += len(searchlist)

        while go:
            length += 1
            nextindxlist = searchlist + 1
            nextindxlist = nextindxlist[nextindxlist < n - 1 - i]  # Remove candidates too close to the end
            nextcandidates = rem_time_series[nextindxlist]
            hitlist = np.abs(nextcandidates - template[length-1]) < tolerance
            searchlist = nextindxlist[hitlist]

            Ntemp[length] += np.sum(hitlist)

            go = hitlist.any() and length < M + 1

    sampen = -np.log(Ntemp[1:] / Ntemp[:-1])
    return sampen


def multiscale_entropy(s:ArrayLike, sample_length:int, tolerance:Optional[Real]=None, maxscale:Optional[int]=None) -> np.ndarray:
    """
    
    calculate the multiscale entropy of the given time series considering
    different time-scales of the time series

    Parameters:
    -----------
    s: array_like,
        the time series for analysis
    sample_length: int,
        bandwidth or group of points
    tolerance: real number,
        tolerance (default = 0.1*std(time_series))
    maxscale: int, optional
        (to check)

    Returns:
    --------
    np.ndarray, the vector containing multiscale entropy

    Reference:
    ----------
        [1] http://en.pudn.com/downloads149/sourcecode/math/detail646216_en.html
    """
    sig_len = len(s)
    if tolerance is None:
        #we need to fix the tolerance at this level. If it remains 'None' it will be changed in call to sample_entropy()
        tolerance = 0.1*np.std(s)
    if maxscale is None:
        maxscale = len(s)

    mse = np.zeros(maxscale)

    for i in range(maxscale):
        b = int(np.fix(sig_len / (i+1)))
        temp = s[0:b*(i+1)].reshape((b, i+1))
        cts = np.mean(temp, axis = 1)
        mse[i] = sample_entropy(cts, sample_length, tolerance)[-1]
    return mse


def permutation_entropy(s:ArrayLike, order:int=3, delay:int=1, normalize:bool=False) -> float:
    """
    
    compute the permutation entropy of the time series `s`

    Parameters
    ----------
    s: array_like,
        the time series for analysis
    order: int,
        order of permutation entropy
    delay: int,
        time delay
    normalize: bool,
        if True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit

    Returns:
    --------
    pe: float,
        permutation entropy

    References
    ----------
    .. [1] Massimiliano Zanin et al. Permutation Entropy and Its Main
        Biomedical and Econophysics Applications: A Review.
        http://www.mdpi.com/1099-4300/14/8/1553/pdf

    .. [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural
        complexity measure for time series.
        http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf
    """
    x = np.array(s)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    temp = np.empty((order, len(x) - (order - 1) * delay))
    for i in range(order):
        temp[i] = x[i * delay:i * delay + temp.shape[1]]
    sorted_idx = temp.T.argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


def multiscale_permutation_entropy(s:ArrayLike, order:int, delay:int, scale:int) -> np.ndarray:
    """
    
    Calculate the Multiscale Permutation Entropy

    Parameters:
    -----------
    s: array_like,
        time series for analysis
    order: int,
        order of permutation entropy
    delay: int,
        time delay
    scale: int,
        scale factor

    Returns:
    --------
    np.ndarray, the vector containing multiscale permutation entropy

    Reference:
    ----------
        [1] Francesco Carlo Morabito et al. Multivariate Multi-Scale Permutation Entropy for
            Complexity Analysis of Alzheimer’s Disease EEG. www.mdpi.com/1099-4300/14/7/1186
        [2] http://www.mathworks.com/matlabcentral/fileexchange/37288-multiscale-permutation-entropy-mpe/content/MPerm.m
    """
    mspe = []
    sig_len = len(s)
    for i in range(scale):
        b = int(np.fix(sig_len / (i+1)))
        temp = s[0:b*(i+1)].reshape((b, i+1))
        coarse_time_series = np.mean(temp, axis = 1)
        pe = permutation_entropy(coarse_time_series, order=order, delay=delay)
        mspe.append(pe)
    return np.array(mspe)


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


def train_test_split_dataframe(df:pd.DataFrame, split_cols:Optional[Union[str,List[str],Tuple[str]]]=None, non_split_cols:Optional[Union[str,List[str],Tuple[str]]]=None, test_ratio:float=0.2, verbose:int=0, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ partly finished,

    make a train-test-split of a DataFrame,
    so that each item in `by` is split by `test_ratio`,
    in a way that
    1. the distributions of certain features are the same in both the train and the test set
    2. subjects with some certain features are not split, preventing data leakage

    Paramters:
    ----------
    df: DataFrame,
        the DataFrame to be split
    split_cols: str, or list or tuple of str, optional,
        name(s) of the column(s) to be treated as the 'labels',
    non_split_cols: str, or list or tuple of str, optional,
        name(s) of the column(s) that should not be split,
        to prevent possible data leakage,
        e.g. `subject_id` which assigns one number to each person,
    test_ratio: float, default 0.2,
        ratio of the test DataFrame, 0.0-1.0
    verbose: int, default 0

    Returns:
    --------
    (df_train, df_test): tuple of DataFrame

    NOTE:
    1. all values of cells in each column of `split_cols` and `non_split_cols` should be hashable
    2. when `split_cols` or `non_split_cols` is a very large list,
       or when some of these columns have large lists of unique values,
       then each 'item' to be split can have very few elements,
       even fewer than 1/`test_ratio`,
       in which cases the test dataframe would be empty

    TODO:
    implement the method of hybrid splitting when `split_cols` and `non_split_cols` are both specified
    """
    _split_cols = split_cols or []
    _non_split_cols = non_split_cols or []
    if isinstance(_split_cols, str):
        _split_cols = [_split_cols]
    else:
        _split_cols = [item for item in _split_cols]
    if isinstance(_non_split_cols, str):
        _non_split_cols = [_non_split_cols]
    else:
        _non_split_cols = [item for item in _non_split_cols]
    
    if not (split_cols or non_split_cols):
        df_train, df_test = _train_test_split_dataframe_naive(
            df=df,
            test_ratio=test_ratio,
            verbose=verbose,
            **kwargs
        )
    elif len(_non_split_cols) == 0:
        df_train, df_test = _train_test_split_dataframe_strafified(
            df=df,
            split_cols=_split_cols,
            test_ratio=test_ratio,
            verbose=verbose,
            **kwargs
        )
    elif len(_split_cols) == 0:
        df_train, df_test = _train_test_split_dataframe_with_nonsplits(
            df=df,
            non_split_cols=_non_split_cols,
            test_ratio=test_ratio,
            verbose=verbose,
            **kwargs
        )
    else:
        df_train, df_test = _train_test_split_dataframe_hybrid(
            df=df,
            split_cols=_split_cols,
            non_split_cols=_non_split_cols,
            test_ratio=test_ratio,
            verbose=verbose,
            **kwargs
        )

    return df_train, df_test


def _train_test_split_dataframe_strafified(df:pd.DataFrame, split_cols:List[str], test_ratio:float=0.2, verbose:int=0, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ref. the function `train_test_split_dataframe`
    """
    df_inspection = df[split_cols]
    for item in split_cols:
        all_entities = df_inspection[item].unique().tolist()
        entities_dict = {e: str(i) for i, e in enumerate(all_entities)}
        df_inspection[item] = df_inspection[item].apply(lambda e:entities_dict[e])

    inspection_col_name = "Inspection" * (max([len(c) for c in split_cols])//10+1)
    df_inspection[inspection_col_name] = df_inspection.apply(
        func=lambda row: "-".join(row.values.tolist()),
        axis=1
    )
    
    item_names = df_inspection[inspection_col_name].unique().tolist()
    item_indices = {
        n: df_inspection.index[df_inspection[inspection_col_name]==n].tolist() for n in item_names
    }
    
    if verbose >= 1:
        print("item_names = {}".format(item_names))
    
    for n in item_names:
        random.shuffle(item_indices[n])
    
    test_indices = []
    for n in item_names:
        item_test_indices = item_indices[n][:int(round(test_ratio*len(item_indices[n])))]
        test_indices += item_test_indices
        if verbose >= 2:
            print("for the item `{}`, len(item_test_indices) = {}".format(n, len(item_test_indices)))
    
    df_test = df.loc[df.index.isin(test_indices)].reset_index(drop=True)
    df_train = df.loc[~df.index.isin(test_indices)].reset_index(drop=True)
    
    return df_train, df_test


def _train_test_split_dataframe_with_nonsplits(df:pd.DataFrame, non_split_cols:List[str], test_ratio:float=0.2, verbose:int=0, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ref. the function `train_test_split_dataframe`
    """
    if len(non_split_cols) > 1:
        raise NotImplementedError("not implemented for the cases where `non_split_cols` > 1")
    tolerance = kwargs.get("tolerance", 0.05) * len(df)
    df_inspection = df[non_split_cols]
    acc = 0.0
    test_indices = []
    all_entities, entities_count = np.unique(df_inspection[non_split_cols[0]].values, return_counts=True)
    pass


def _train_test_split_dataframe_naive(df:pd.DataFrame, test_ratio:float=0.2, verbose:int=0, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ref. the function `train_test_split_dataframe`
    """
    n_cols = len(df)
    indices = list(range(n_cols))
    random.shuffle(indices)
    test_indices = indices[:int(round(test_ratio*n_cols))]
    df_test = df.loc[df.index.isin(test_indices)].reset_index(drop=True)
    df_train = df.loc[~df.index.isin(test_indices)].reset_index(drop=True)

    return df_train, df_test


def _train_test_split_dataframe_hybrid(df:pd.DataFrame, split_cols:List[str], non_split_cols:List[str], test_ratio:float=0.2, verbose:int=0, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ref. the function `train_test_split_dataframe`
    """
    raise NotImplementedError
