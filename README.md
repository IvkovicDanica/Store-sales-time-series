# Store Sales Forecasting

This repository contains the code and methodology used for forecasting sales of product families at Favorita stores located in Ecuador. Our team participated in a Kaggle competition and achieved 9th place using two ensembles of LightGBM (LGBM) models and the `darts` library for time series forecasting.

## Dataset

The dataset used in this project is provided by Corporación Favorita through Kaggle. It includes daily sales data for a variety of product families across multiple stores in Ecuador. 

- [Kaggle Competition Dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

## Methodology

We employed the `darts` library and LightGBM (LGBM) models for time series forecasting. The key steps in our methodology include:

1. **Data Preprocessing**: Handling missing values, outlier detection, and feature engineering.
2. **Model Selection**: Using two ensembles of four LGBMs each:
   - One ensemble trained on the full dataset.
   - Another ensemble trained on data from 2015 onwards.
3. **Training and Validation**: Splitting the data into training and validation sets, and tuning the models for optimal performance.
4. **Evaluation**: Using RMSLE to evaluate model performance.
