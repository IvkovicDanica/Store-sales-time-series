import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

############################################### PLOTING FUNCTIONS ###############################################

def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta(days=365) / pd.Timedelta(days=1)
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(25, 10))
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


def plot_period_mean(df_original: pd.DataFrame, target_variable: str, period: str='M'):
    if period == 'M':
        temp = df_original.set_index('date').resample(period)[[target_variable]].mean().reset_index()
        temp['year'] = temp.date.dt.year
    elif period == 'DW':
        temp = df_original.copy()
        temp["year"] = temp.date.dt.year
        temp["dayofweek"] = temp.date.dt.dayofweek
        temp = temp.groupby(["year", "dayofweek"])[[target_variable]].mean().reset_index()
    else:
        raise ValueError('Not valid value for period, only M and DW supported')
    
    # Create a figure with a larger size
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot each year's transactions
    for year, data in temp.groupby('year'):
        if period == 'M':
            ax.plot(data['date'], data[target_variable], label=f'Year {year}')
        else:
            dayofweek_mean = data.groupby('dayofweek')[target_variable].mean()
            ax.plot(dayofweek_mean.index, dayofweek_mean.values, label=f'Year {year}')
            
    # Set the title and labels
    title = f'Monthly Average {target_variable}' if period == 'M' else f'{target_variable} by Day of the Week' 
    xlabel = 'Date' if period == 'M' else 'Day of the Week'
    ax.set_title(f'{title}', fontsize=20)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel(f'{xlabel}', fontsize=15)
    ax.legend(title='Year', fontsize=12, title_fontsize=14)

    if period == 'DW':
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=45, fontsize=12)

    
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=12)
    # Increase the y-axis label size
    plt.yticks(fontsize=12)

    plt.show()
    


def plot_boxplot(palette, x, y, hue, ax, title):
    sns.set_theme(style="ticks", palette=palette)
    ax = sns.boxplot(x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title, fontsize=18)
    

def get_seasonality_trend_overview(df_original: pd.DataFrame, target_variable: str, title_name: str=None):
    if not title_name:
        title_name = target_variable
    df_sesonality = df_original.copy()
    df_sesonality = df_sesonality.groupby('date').mean()[[target_variable]].reset_index()
    #create month and year variables from date colum
    df_sesonality['date'] = pd.to_datetime(df_sesonality['date'])
    df_sesonality['year'] = df_sesonality['date'].dt.year
    df_sesonality['month'] = df_sesonality['date'].dt.month
    df_sesonality['day'] = df_sesonality['date'].dt.day
    df_sesonality['dayofweek'] = df_sesonality['date'].dt.day_name()

    fig, ax = plt.subplots(figsize=(15, 6))

    palette = sns.color_palette("ch:2.5,-.2,dark=.3", 10)
    sns.lineplot(x = df_sesonality['month'], y = df_sesonality[target_variable], hue=df_sesonality['year'])
    ax.set_title(f'Seasonal plot of {title_name}', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax.set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))
    ax.set_ylabel(f'{title_name}', fontsize = 16, fontdict=dict(weight='bold'))


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17, 6))

    # Year-wise box plot
    sns.boxplot(x=df_sesonality['year'], y=df_sesonality[target_variable], palette="turbo",ax=ax[0])
    ax[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[0].set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))
    ax[0].set_ylabel(f'{title_name} by Year', fontsize = 16, fontdict=dict(weight='bold'))

    # Month-wise box plot
    sns.boxplot(x=df_sesonality['month'], y=df_sesonality[target_variable], palette="Pastel2", ax=ax[1])
    ax[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[1].set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))
    ax[1].set_ylabel(f'{title_name} by Month', fontsize = 16, fontdict=dict(weight='bold'))

    #plot boxplots for every day
    fig = plt.figure(figsize=(17,7))
    sns.boxplot(x=df_sesonality['day'], y=df_sesonality[target_variable], palette="YlOrRd")

    #plot boxplots for every day
    fig = plt.figure(figsize=(17,7))
    sns.boxplot(x=df_sesonality['dayofweek'], y=df_sesonality[target_variable], palette="GnBu",
                order=['Monday', 'Tuesday', 'Wednesday', 
                        'Thursday', 'Friday', 'Saturday','Sunday'])
    
    
    
############################################### FEATURE ENGINEERING FUNCTIONS ###############################################    

def create_date_features(df: pd.DataFrame):
    df['month'] = df.date.dt.month.astype('int8')
    df['day_of_month'] = df.date.dt.day.astype('int8')
    df['day_of_year'] = df.date.dt.dayofyear.astype('int16')
    df['week_of_month'] = (df.date.apply(lambda d: (d.day-1) // 7 + 1)).astype('int8')
    df['week_of_year'] = (df.date.dt.isocalendar().week).astype('int8')
    df['day_of_week'] = (df.date.dt.dayofweek + 1).astype('int8') # since our transactions/sales depend on day of the week this feature will capture seasonality
    df['year'] = df.date.dt.year.astype('int32')
    df['is_wknd'] = (df.date.dt.weekday // 4).astype('int8')
    df['quarter'] = df.date.dt.quarter.astype('int8')
    df['is_month_start'] = df.date.dt.is_month_start.astype('int8')
    df['is_month_end'] = df.date.dt.is_month_end.astype('int8')
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype('int8')
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype('int8')
    df['is_year_start'] = df.date.dt.is_year_start.astype('int8')
    df['is_year_end'] = df.date.dt.is_year_end.astype('int8')
    df["date_index"] = df.date.factorize()[0]
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df['season'] = np.where(df.month.isin([12,1,2]), 0, 1)
    df['season'] = np.where(df.month.isin([6,7,8]), 2, df['season'])
    df['season'] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df['season'])).astype('int8')
    return df


def create_work_related_features(df: pd.DataFrame):
    df['workday'] = np.where((df.local_holidays == 1) | (df.national_holidays==1) | (df.regional_holidays==1) | (df['day_of_week'].isin([6,7])), 0, 1)
    df['wageday'] = pd.Series(np.where((df['is_month_end'] == 1) | (df['day_of_month'] == 15), 1, 0)).astype('int8')
    return df



############################################### ZERO FORECASTING ###############################################
def num_trailing_zeros(series: pd.Series):
    '''
    Checks number of trailing zeros in a series
    ex. [1, 2, 3, 0, 0, 0] -> 3
    '''
    # finds indices of non-zero elements and returns that array
    nz_idx = np.where(series!=0)[0]
    # Checks if the array of non-zero indices is empty. If it is, it means all elements in the series are zero 
    if len(nz_idx) == 0:
        return len(series)
    else:
        # If the array of non-zero indices is not empty, the number of trailing zeros is calculated as the length of the series minus the last non-zero index minus one
        return len(series) - nz_idx[-1] - 1
    

def num_leading_zeros(series: pd.Series):
    '''
    Checks number of leading zeros in a series
    ex. [0,0,0,1,2] -> 3
    '''
    # finds indices of non-zero elements and returns that array
    nz_idx = np.where(series != 0)[0]
    # if that array is empty then there are same amount of zeros as the length of a series
    if len(nz_idx) == 0:
        return len(series)
    else:
        # returns index of a first non-zero element
        return nz_idx[0]


if __name__ == '__main__':
    pass