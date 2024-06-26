import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import periodogram
import numpy as np
import pandas as pd
import scipy.stats as stats


####################################### PLOTING FUNCTIONS ########################################

def plot_periodogram(ts, detrend='linear', ax=None):
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
    
    
def plot_zeros(data_to_plot, name):
    fig = plt.figure(figsize=(18,6))
    ax0 = sns.boxplot(data=data_to_plot, x="store_nbr", y=name, color="blue", ax=fig.add_subplot(211))
    plt.title(f"{name} within each store")

    # by family
    ax1 = sns.boxplot(data=data_to_plot, x="family", y=name, color="yellow", ax=fig.add_subplot(212))
    plt.title(f"{name} within each family")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()


def plot_boxplots(data : pd.DataFrame, y : str):

    plot_kwargs = {
        "palette": ["red", "green"],
        "linewidth": 2,
        "flierprops": {"alpha": 0.2},
        "orient": "h",
    }
    
    fig = plt.figure(figsize=(18, 20))

    sns.boxplot(
        data=data, #saturday
        y=y,
        x="sales_scaled",
        ax=fig.add_subplot(531),
        **plot_kwargs,
    )
    plt.yticks([0, 1], ["no", "yes"])
        
    plt.suptitle(f"Distribution of {y}")
    plt.tight_layout()
    plt.show()

  
def plot_correlation(col1: pd.Series, col2: pd.Series):
    # calculate the correlation
    correlation = col1.corr(col2)
    print(f"Correlation between {col1.name} and {col2.name}: {correlation}")

    plt.figure(figsize=(14, 6))

    # plot scatter plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=col1, y=col2)
    plt.title(f'Scatter plot of {col1.name} vs {col2.name}')
    plt.xlabel(col1.name)
    plt.ylabel(col2.name)

    # plot heatmap
    plt.subplot(1, 2, 2)
    corr_matrix = pd.DataFrame({col1.name: col1, col2.name: col2}).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
    plt.title('Correlation heatmap')

    plt.tight_layout()
    plt.show()
  

################################### A/B TESTING ###############################################

def AB_test_local_holidays(dataframe, group, target, holidays):
    s = holidays['name']
    city = holidays[s == group].city.values[0]
    #splitting groups
    groupA = dataframe[(dataframe['city']==city)&(dataframe[group]==1)&(dataframe['national_holidays']==0)&(dataframe['regional_holidays']==0)][target] #holiday
    groupB = dataframe[(dataframe['city']==city)&(dataframe[group]==0)&(dataframe['national_holidays']==0)&(dataframe['regional_holidays']==0)][target] #not holiday
    leveneTest_p = stats.levene(groupA, groupB)[1]

    if leveneTest_p<0.05:
        p = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
    else:
        p = stats.ttest_ind(groupA, groupB, equal_var=True)[1]

    group = [group]
    p = [p]

    AB = pd.DataFrame({
    "Feature": group,
    "p-value": p,
    #"Test": np.where((np.array(pA) == False) & (np.array(pB) == False), "t-Test (p)", "Mann-Whitney U (nonp)"),
    "Hypothesis": np.where(np.array(p) >= 0.05, "Fail to Reject H0", "Reject H0"),
    "Comment": np.where(np.array(p) >= 0.05, "A/B groups are similar", "A/B groups are not similar"),
    "GroupA_mean": np.mean(groupA),
    "GroupB_mean": np.mean(groupB),
    "GroupA_median": np.median(groupA),
    "GroupB_median": np.median(groupB)
    })
    return AB  

  
def AB_test_national_holidays(dataframe, group, target):
    #splitting groups
    groupA = dataframe[(dataframe[group] == 1)][target] #holiday
    groupB = dataframe[(dataframe[group] == 0)&(dataframe['national_holidays']==0)][target] #not holiday
    leveneTest_p = stats.levene(groupA, groupB)[1]

    if leveneTest_p<0.05:
        p = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
    else:
        p = stats.ttest_ind(groupA, groupB, equal_var=True)[1]

    group = [group]
    p = [p]

    AB = pd.DataFrame({
    "Feature": group,
    "p-value": p,
    #"Test": np.where((np.array(pA) == False) & (np.array(pB) == False), "t-Test (p)", "Mann-Whitney U (nonp)"),
    "Hypothesis": np.where(np.array(p) >= 0.05, "Fail to Reject H0", "Reject H0"),
    "Comment": np.where(np.array(p) >= 0.05, "A/B groups are similar", "A/B groups are not similar"),
    "GroupA_mean": np.mean(groupA),
    "GroupB_mean": np.mean(groupB),
    "GroupA_median": np.median(groupA),
    "GroupB_median": np.median(groupB)
    })
    return AB


def AB_test_regional_holidays(dataframe, group, target, holidays):
    s = holidays['name']
    state = holidays[s == group].state.values[0]
    groupA = dataframe[(dataframe['state']==state)&(dataframe[group]==1)&(dataframe['national_holidays']==0)][target] #holiday
    groupB = dataframe[(dataframe['state']==state)&(dataframe[group]==0)&(dataframe['national_holidays']==0)][target] #not holiday
    leveneTest_p = stats.levene(groupA, groupB)[1]

    if leveneTest_p<0.05:
        p = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
    else:
        p = stats.ttest_ind(groupA, groupB, equal_var=True)[1]

    group = [group]
    p = [p]

    AB = pd.DataFrame({
    "Feature": group,
    "p-value": p,
    #"Test": np.where((np.array(pA) == False) & (np.array(pB) == False), "t-Test (p)", "Mann-Whitney U (nonp)"),
    "Hypothesis": np.where(np.array(p) >= 0.05, "Fail to Reject H0", "Reject H0"),
    "Comment": np.where(np.array(p) >= 0.05, "A/B groups are similar", "A/B groups are not similar"),
    "GroupA_mean": np.mean(groupA),
    "GroupB_mean": np.mean(groupB),
    "GroupA_median": np.median(groupA),
    "GroupB_median": np.median(groupB)
    })
    return AB
  
################################### FEATURE ENGINEERING ###################################

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
  
  
if __name__ == '__main__':
    pass