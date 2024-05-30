import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def plot_period_mean(df_original: pd.DataFrame, period: str='M'):
    if period == 'M':
        temp = df_original.set_index('date').resample(period).transactions.mean().reset_index()
        temp['year'] = temp.date.dt.year
    elif period == 'DW':
        temp = df_original.copy()
        temp["year"] = temp.date.dt.year
        temp["dayofweek"] = temp.date.dt.dayofweek
        temp = temp.groupby(["year", "dayofweek"]).transactions.mean().reset_index()
    else:
        raise ValueError('Not valid value for period, only M and DW supported')
    
    # Create a figure with a larger size
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot each year's transactions
    for year, data in temp.groupby('year'):
        if period == 'M':
            ax.plot(data['date'], data['transactions'], label=f'Year {year}')
        else:
            dayofweek_mean = data.groupby('dayofweek')['transactions'].mean()
            ax.plot(dayofweek_mean.index, dayofweek_mean.values, label=f'Year {year}')
            
    # Set the title and labels
    title = 'Monthly Average Transactions' if period == 'M' else 'Transactions by Day of the Week' 
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

    sns.boxplot(x=df_sesonality['year'], y=df_sesonality[target_variable], palette="turbo",ax=ax[0])
    ax[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[0].set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))
    ax[0].set_ylabel(f'{title_name} by Year', fontsize = 16, fontdict=dict(weight='bold'))

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
    
    
if __name__ == '__main__':
    pass