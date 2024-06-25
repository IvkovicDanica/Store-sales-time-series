import numpy as np
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler, InvertibleMapper, StaticCovariatesTransformer
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller
from sklearn.preprocessing import OneHotEncoder
from darts.models.filtering.moving_average_filter import MovingAverageFilter
from tqdm import tqdm


def get_pipeline(static_covs_transform=False, log_transform=False):
    """
    Creates a data preprocessing pipeline for time series data, which includes filling 
    missing values, optional one-hot encoding of static covariates, optional log transformation, 
    and scaling of the data.

    Parameters:
    ----------
        static_covs_transform (bool, optional): 
            If True, one-hot encodes static covariates. Defaults to False.
        log_transform (bool, optional): 
            If True, applies a log transformation to the data. Defaults to False.

    Returns:
    ----------
        Pipeline: A pipeline object that chains together the specified data transformation 
        steps including missing value interpolation, optional one-hot encoding, optional log 
        transformation, and scaling.
        
    The pipeline includes the following steps:
        1. Missing Values Filler: Fills missing values in the time series using interpolation.
        2. (Optional) Static Covariates Transformer: One-hot encodes static covariates if 
           `static_covs_transform` is set to True.
        3. (Optional) Log Transformer: Applies a log transformation to the time series data if 
           `log_transform` is set to True.
        4. Scaler: Rescales the time series data.
    """
    # contains all steps of our pipeline
    lst = []
    
    # fills missing values, uses interpolation
    # n_jobs = -1 -> allows for parallel processing using all available CPU cores
    filler = MissingValuesFiller(n_jobs=-1)
    lst.append(filler)
    
    # one hot encoding static covariates
    if static_covs_transform:
        static_covs_transformer = StaticCovariatesTransformer(
            transformer_cat=OneHotEncoder(), 
            n_jobs=-1,
        )
        lst.append(static_covs_transformer)

    # perform log transformation on sales
    if log_transform:
        log_transformer = InvertibleMapper(
            fn=np.log1p,
            inverse_fn=np.expm1,
            n_jobs=-1,
        )
        lst.append(log_transformer)

    # rescale time series
    scaler = Scaler()
    lst.append(scaler)

    pipeline = Pipeline(lst)
    return pipeline


def get_target_series(data, static_cols,  train_date_end="2017-08-15", log_transform=True):
    """
    Extracts and preprocesses target time series data for each family from the input dataframe.

    Parameters:
    ----------
        data (DataFrame): The dataframe containing time series data, including columns 'date', 'sales', 
                          'store_nbr', and 'family'.
        static_cols (list): List of column names representing static covariates to be included in 
                            the time series transformation.
        train_date_end (str, optional): End date for the training data. Data after this date is excluded. 
                                        Defaults to "2017-08-15".
        log_transform (bool, optional): If True, applies a log transformation to the sales data. Defaults to True.

    Returns:
    ----------
        tuple: A tuple containing three dictionaries:
            - target_dict: Dictionary where keys are family names and values are lists of preprocessed 
                           time series arrays of sales data by store.
            - pipe_dict: Dictionary where keys are family names and values are the corresponding 
                         transformation pipelines used for preprocessing.
            - id_dict: Dictionary where keys are family names and values are lists of dictionaries, 
                       each containing 'store_nbr' and 'family' identifiers corresponding to each time series.

    The function iterates over each unique family in the input dataframe, extracts time series data up to 
    the specified `train_date_end`, creates a transformation pipeline using `get_pipeline`, and transforms 
    the time series data. It then records the transformed time series data, transformation pipelines, and 
    identifiers for each family.
    """
    target_dict = {} # key is family, value is array of time series of sales by stores
    pipe_dict = {} # key is family, value is pipeline
    id_dict = {} # key is family, value is pair of store and family

    for fam in tqdm(data.family.unique(), desc="Extracting target series"):
        # using train and splitting by family
        df = data[(data.date.le(train_date_end)) & (data.family.eq(fam))] 
        
        # initialize transformation pipeline for target series
        pipe = get_pipeline(static_covs_transform=True, log_transform=log_transform)
        
        # The TimeSeries.from_group_dataframe method is used to create a list of time series objects from the dataframe, 
        # grouped by store_nbr and including static covariates specified in static_cols
        target = TimeSeries.from_group_dataframe(
            df=df,
            time_col="date",
            value_cols="sales",
            group_cols="store_nbr",
            static_cols=static_cols,
        )
        
        # record identity of each target series
        target_id = [{"store_nbr": t.static_covariates.store_nbr, "family": fam} # pair family store
                     for t in target]
        id_dict[fam] = target_id

        # apply transformations
        target = pipe.fit_transform(target)
        target_dict[fam] = [t.astype(np.float32) for t in target]

        pipe_dict[fam] = pipe[2:]  # without MissingValuesFiller and OHEnc
        
    return target_dict, pipe_dict, id_dict


def get_covariates(data, past_cols, future_cols, future_ma_cols=None, future_window_sizes=[7, 28]):
    """
    Extracts past and future covariates from the input dataframe for each family.

    Parameters:
    ----------
        data (DataFrame): The dataframe containing covariate data, including columns 'date', 
                          'store_nbr', and specified columns for past and future covariates.
        past_cols (list): List of column names representing past covariates to be extracted.
        future_cols (list): List of column names representing future covariates to be extracted.
        future_ma_cols (list, optional): List of column names for which moving average features are computed.
                                         Defaults to None.
        future_window_sizes (list, optional): List of integers representing window sizes for the moving average 
                                              computation. Defaults to [7, 28].

    Returns:
    ----------
        tuple: A tuple containing two dictionaries:
            - past_dict: Dictionary where keys are family names and values are lists of arrays 
                         representing past covariates for each store.
            - future_dict: Dictionary where keys are family names and values are lists of arrays 
                           representing future covariates (including potential moving averages) 
                           for each store.

    The function iterates over each unique family in the input dataframe, extracts past and future 
    covariates up to a specified date, applies transformation pipelines, and computes moving averages 
    for specified columns if requested. It then stores the transformed covariates in dictionaries 
    indexed by family names.
    """
    past_dict = {} # key is family, value is array of time series of transactions by stores
    future_dict = {} # key is family, value is array of arrays of time series of future covariate by stores for all future covariates
    
    # initialize transformation pipeline for covariates
    covs_pipe = get_pipeline()

    for fam in tqdm(data.family.unique(), desc="Extracting covariates"):
        # filter data for each model
        df = data[data.family.eq(fam)]
        
        # extract past covariates
        past_covs = TimeSeries.from_group_dataframe(
            df=df[df.date.le("2017-08-15")],
            time_col="date",
            value_cols=past_cols,
            group_cols="store_nbr",
        )
        past_covs = [p.with_static_covariates(None) for p in past_covs]
        past_covs = covs_pipe.fit_transform(past_covs)
        
        past_dict[fam] = [p.astype(np.float32) for p in past_covs]

        # extract future covariates
        future_covs = TimeSeries.from_group_dataframe(
            df=df,
            time_col="date",
            value_cols=future_cols,
            group_cols="store_nbr",
        )
        future_covs = [f.with_static_covariates(None) for f in future_covs]
        future_covs = covs_pipe.fit_transform(future_covs)
        
        if future_ma_cols is not None:
            for size in future_window_sizes:
                ma_filter = MovingAverageFilter(window=size)
                old_names = [f"rolling_mean_{size}_{col}" for col in future_ma_cols]
                new_names = [f"{col}_ma{size}" for col in future_ma_cols]
                future_ma_covs = [
                    ma_filter.filter(f[future_ma_cols]).with_columns_renamed(old_names, new_names) 
                    for f in future_covs
                ]
                future_covs = [f.stack(f_ma) for f, f_ma in zip(future_covs, future_ma_covs)]
        
        future_dict[fam] = [f.astype(np.float32) for f in future_covs]
            
    return past_dict, future_dict


if __name__ == "__main__":
    pass