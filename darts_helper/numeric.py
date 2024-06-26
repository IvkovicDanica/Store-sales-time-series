import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error


def rmsle(valid, pred):
    """
    Computes the Root Mean Squared Logarithmic Error (RMSLE) between the valid and predicted time series.

    Parameters:
    ----------
    valid : list of TimeSeries
        A list of validation TimeSeries objects.
    pred : list of TimeSeries
        A list of predicted TimeSeries objects.

    Returns:
    -------
    float
        The mean RMSLE value computed across all time series in the validation set.
    """
    valid_df = pd.concat([ts.pd_dataframe() for ts in valid], axis=1)
    pred_df = pd.concat([ts.pd_dataframe() for ts in pred], axis=1)

    # calculate RMSLE for each pair of valid and predicted values
    rmsle_values = [mean_squared_log_error(valid_df[col], pred_df[col], squared=False) for col in valid_df.columns]

    # calculate the mean of RMSLE values of all series of that family
    return np.mean(rmsle_values)


def clip(array):
    """
    Changes negative values of an array to zeroes.
    
    Parameters:
    ----------
    array: The input array to be clipped.

    Returns:
    ---------
    np.ndarray: The clipped array with no negative values.
    """
    return np.clip(array, a_min=0., a_max=None)


if __name__ == "__main__":
    pass