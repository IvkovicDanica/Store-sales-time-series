import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings


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
    sns.lineplot(x = df_sesonality['month'], y = df_sesonality[target_name], hue=df_sesonality['year'])
    ax.set_title(f'Seasonal plot of {title_name}', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax.set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))
    ax.set_ylabel(f'{title_name}', fontsize = 16, fontdict=dict(weight='bold'))


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17, 6))

    sns.boxplot(x=df_sesonality['year'], y=df_sesonality[target_name], palette="turbo",ax=ax[0])
    ax[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[0].set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))
    ax[0].set_ylabel(f'{title_name} by Year', fontsize = 16, fontdict=dict(weight='bold'))

    sns.boxplot(x=df_sesonality['month'], y=df_sesonality[target_name], palette="Pastel2", ax=ax[1])
    ax[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[1].set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))
    ax[1].set_ylabel(f'{title_name} by Month', fontsize = 16, fontdict=dict(weight='bold'))

    #plot boxplots for every day
    fig = plt.figure(figsize=(17,7))
    sns.boxplot(x=df_sesonality['day'], y=df_sesonality[target_name], palette="YlOrRd")

    #plot boxplots for every day
    fig = plt.figure(figsize=(17,7))
    sns.boxplot(x=df_sesonality['dayofweek'], y=df_sesonality[target_name], palette="GnBu",
                order=['Monday', 'Tuesday', 'Wednesday', 
                        'Thursday', 'Friday', 'Saturday','Sunday'])
    
    
if __name__ == '__main__':
    pass