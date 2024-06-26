import numpy as np
import pandas as pd
from darts import TimeSeries
from tqdm.notebook import tqdm_notebook
from dataclasses import dataclass
from darts_helper.numeric import rmsle, clip
from darts_helper.models import get_models


@dataclass
class Trainer:
    """
    A class used to train, validate, and generate forecasts using various models.

    Attributes:
    ----------
    target_dict : dict
        A dictionary containing the target time series data for different families.
    pipe_dict : dict
        A dictionary containing the transformation pipelines for different families.
    id_dict : dict
        A dictionary containing the ID mappings for different families.
    past_dict : dict
        A dictionary containing the past covariates for different families.
    future_dict : dict
        A dictionary containing the future covariates for different families.
    forecast_horizon : int
        The forecast horizon for generating predictions.
    folds : int
        The number of folds for cross-validation.
    zero_fc_window : int
        The window size to check for all zero values in the time series.
    static_covs : str
        The static covariates.
    past_covs : str
        The past covariates.
    future_covs : str
        The future covariates.
    """
    target_dict: dict 
    pipe_dict: dict 
    id_dict: dict 
    past_dict: dict 
    future_dict: dict 
    forecast_horizon: int 
    folds: int
    zero_fc_window: int 
    static_covs: str
    past_covs: str
    future_covs: str
    models: list
    
    
    def train_valid_split(self, target, length):
        """
        Splits the target time series into training and validation sets.

        Parameters:
        ----------
        target : list of TimeSeries
            The target time series data.
        length : int
            The length for validation split.

        Returns:
        -------
        tuple
            A tuple containing the training and validation sets.
        """
        train = [timeseries[:-length] for timeseries in target]
        
        # ensures that window of forecasting is of length forecast_horizon or smaller
        valid_end_idx = -length + self.forecast_horizon
        
        if valid_end_idx >= 0:
            valid_end_idx = None
        
        valid = [t[-length:valid_end_idx] for t in target]
        return train, valid
    
    
    def predict(self, train, pipe, past_covs, future_covs, drop_before, train_cycle=True):
        """
        Generates forecasts using multiple models, applies inverse transformations, and performs ensemble averaging.

        Parameters:
        ----------
            train (list): List of TimeSeries objects representing the training data for each store.
            pipe (Pipeline): Transformation pipeline to apply inverse transformations on the forecasts.
            past_covs (list): List of TimeSeries objects representing past covariates for each store.
            future_covs (list): List of TimeSeries objects representing future covariates for each store.
            drop_before (str or None): If provided, drops data before this date from the training series. 
                                    The date should be in 'YYYY-MM-DD' format.
            train_cycle (bool, optional): If True, fits the model on the training data. Defaults to True.

        Returns:
        -------
            tuple: A tuple containing two elements:
                - pred_list (list): List of lists of TimeSeries objects. Each sublist contains the forecasts 
                                    from one of the models for each store.
                - ens_pred (list): List of TimeSeries objects representing the ensemble averaged forecasts 
                                for each store.
        """
        if drop_before is not None: 
            date = pd.Timestamp(drop_before) - pd.Timedelta(days=1) 
            # train without specifed dates
            train = [t.drop_before(date) for t in train]
             
        # inputs for a model
        inputs = {
            "series": train,
            "past_covariates": past_covs,
            "future_covariates": future_covs,
        }

        # generates validation dates and all zero values for those dates
        zero_pred = pd.DataFrame({ 
            "date": pd.date_range(train[0].end_time(), periods=self.forecast_horizon+1)[1:],
            "sales": np.zeros(self.forecast_horizon),
        })
        
        # transforming that df to time series
        zero_pred = TimeSeries.from_dataframe( 
            df=zero_pred,
            time_col="date",
            value_cols="sales",
        )
        
        pred_list = []
        ens_pred = [0 for _ in range(len(train))] # zero for every store 
        
        for m in self.models:
            # fit training data to model
            if train_cycle:
                m.fit(**inputs)

            # generate forecasts
            pred = m.predict(n=self.forecast_horizon, **inputs, show_warnings=False)
            # apply inverse transformations
            pred = pipe.inverse_transform(pred)

            for j in range(len(train)):
                # if there is all zeros in j time series in the last specifed period of time predict zeros
                if train[j][-self.zero_fc_window:].values().sum() == 0:
                    pred[j] = zero_pred
            
            # clip negative forecasts to 0s
            pred = [p.map(clip) for p in pred]
            pred_list.append(pred)
            
            # ensemble averaging
            for j in range(len(ens_pred)): # 54
                ens_pred[j] += pred[j] / len(self.models) 

        return pred_list, ens_pred
    
    
    def train(self, model_names, model_configs, drop_before=None):
        """
        Trains and validates models on the provided time series data, performs ensemble averaging, and prints the results.

        Parameters:
        ----------
            model_names (list): List of model names to instantiate. Supported names include 'lr' for 
                                LinearRegressionModel, 'lgbm' for LightGBMModel, and 'xgb' for XGBModel.
            model_configs (list of dict): List of dictionaries containing configurations for each model 
                                        specified in model_names. Each dictionary should contain parameters 
                                        specific to the corresponding model.
            drop_before (str or None): If provided, drops data before this date from the training series. 
                                    The date should be in 'YYYY-MM-DD' format. Defaults to None.
        """
        
        # helper value to align printed text below
        longest_len = len(max(self.target_dict.keys(), key=len)) #33 kao broj prodavnica
        
        # store metric values for each model
        model_metrics_history = []
        ens_metric_history = []
        
        for fam in tqdm_notebook(self.target_dict, desc="Performing validation"):
            target = self.target_dict[fam]
            pipe = self.pipe_dict[fam]
            past_covs = self.past_dict[fam]
            future_covs = self.future_dict[fam]
            
            # record average metric value over all folds
            model_metrics = []
            ens_metric = 0
            
            for j in range(self.folds):    # folds = 1
                # perform train-validation split and apply transformations
                length = (self.folds - j) * self.forecast_horizon # 16
                train, valid = self.train_valid_split(target, length) 
                valid = pipe.inverse_transform(valid) 

                # generate forecasts and compute metric
                self.models = get_models(model_names, model_configs)
                pred_list, ens_pred = self.predict(train, pipe, past_covs, future_covs, drop_before) 
                metric_list = [rmsle(valid, pred) / self.folds for pred in pred_list]
                model_metrics.append(metric_list)
                if len(self.models) > 1:
                    ens_metric_fold = rmsle(valid, ens_pred) / self.folds
                    ens_metric += ens_metric_fold
                
            # store final metric value for each model
            model_metrics = np.sum(model_metrics, axis=0)
            model_metrics_history.append(model_metrics)
            ens_metric_history.append(ens_metric)
            
            # print metric value for each family
            print(
                fam,
                " " * (longest_len - len(fam)),
                " | ",
                " - ".join([f"{model}: {metric:.5f}" for model, metric in zip(model_names, model_metrics)]),
                f" - ens: {ens_metric:.5f}" if len(self.models) > 1 else "",
                sep="",
            )
            
        # print overall metric value
        print(
            "Average RMSLE | "
            + " - ".join([f"{model}: {metric:.5f}" 
                          for model, metric in zip(model_names, np.mean(model_metrics_history, axis=0))])
            + (f" - ens: {np.mean(ens_metric_history):.5f}" if len(self.models) > 1 else ""),
        )
        
    def ensemble_predict(self, drop_before=None):
        """
        Generates ensemble forecasts for all families and stores, and combines them into a single DataFrame.
        
        Parameters:
        ----------
            drop_before (str or None): If provided, drops data before this date from the training series. 
                                    The date should be in 'YYYY-MM-DD' format. Defaults to None.
        """        
        forecasts = []
        for fam in tqdm_notebook(self.target_dict.keys(), desc="Generating forecasts"):
            target = self.target_dict[fam]
            pipe = self.pipe_dict[fam]
            target_id = self.id_dict[fam]
            past_covs = self.past_dict[fam]
            future_covs = self.future_dict[fam]
            
            _, ens_pred = self.predict(self.models, target, pipe, past_covs, future_covs, drop_before, train_cycle=False)
            ens_pred = [p.pd_dataframe().assign(store_nbr=i["store_nbr"]["sales"], family=i["family"]) for p, i in zip(ens_pred, target_id)]
            ens_pred = pd.concat(ens_pred, axis=0)
            forecasts.append(ens_pred)
            
        # combine all forecasts into one dataframe
        forecasts = pd.concat(forecasts, axis=0)
        forecasts = forecasts.rename_axis(None, axis=1).reset_index(names="date")
        
        return forecasts