import pandas as pd
from scipy.ndimage import median_filter

def median_smoothing(X, window_size,full=True):
    """
    Applying median filter to a time series.
    Initial window values can be other use the all available previous values for smoothing,
    or use the initial window values (resulting to same smoothed value)
    :param X: time series data
    :param window_size: The window of the filter
    :return: filtered time-series same shape as X.
    """

    smoothed_series = median_filter(X, size=window_size)
    if full:
        for i in range(window_size):
            smoothed_series[i]=smoothed_series[window_size]
    return smoothed_series


def mean_smoothing(X, window_size,full=True):
    """
    Applying mean filter to a time series.
    Initial window values can be other use the all available previous values for smoothing,
    or use the initial window values (resulting to same smoothed value)

    Note that use of larger window create a shift in the anomaly scores


    :param X: time series data
    :param window_size: The window of the filter
    :return: filtered time-series same shape as X.
    """
    X_df=pd.Series(X)
    smoothed_series = X_df.rolling(window=window_size, center=False, min_periods=1).mean()
    if full:
        for i in range(window_size):
            smoothed_series[i]=smoothed_series[window_size]
    return smoothed_series

def max_smoothing(X, window_size,full=True):
    """
    Applying max filter to a time series.
    Initial window values can be other use the all available previous values for smoothing,
    or use the initial window values (resulting to same smoothed value)
    :param X: time series data
    :param window_size: The window of the filter
    :return: filtered time-series same shape as X.
    """
    X_df=pd.Series(X)
    smoothed_series = X_df.rolling(window=window_size, center=False, min_periods=1).max()
    if full:
        for i in range(window_size):
            smoothed_series[i]=smoothed_series[window_size]
    return smoothed_series

def min_smoothing(X, window_size,full=True):
    """
    Applying min filter to a time series.
    Initial window values can be other use the all available previous values for smoothing,
    or use the initial window values (resulting to same smoothed value)
    :param X: time series data
    :param window_size: The window of the filter
    :return: filtered time-series same shape as X.
    """
    X_df=pd.Series(X)
    smoothed_series = X_df.rolling(window=window_size, center=False, min_periods=1).min()
    if full:
        for i in range(window_size):
            smoothed_series[i]=smoothed_series[window_size]
    return smoothed_series



def ewma_smoothing(X,alpha=0.1):
    """
    Applying ewma filter to a time series.
    Initial window values can be other use the all available previous values for smoothing,
    or use the initial window values (resulting to same smoothed value)
    :param X: time series data
    :param alpha: smoothing factor alpha
    :param window_size: The window of the filter
    :return: filtered time-series same shape as X.
    """
    X_df=pd.Series(X)
    smoothed_series = X_df.ewm(alpha=alpha, adjust=False).mean()
    return smoothed_series


