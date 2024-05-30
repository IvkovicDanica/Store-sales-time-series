import numpy as np
import pandas as pd
import re

train=pd.read_csv(r'originalni_datasetovi\train.csv',parse_dates=['date'])
test=pd.read_csv(r'originalni_datasetovi\test.csv', parse_dates=['date'])
holidays=pd.read_csv(r'originalni_datasetovi\holidays_events.csv',parse_dates=['date'])
oil=pd.read_csv(r'originalni_datasetovi\oil.csv', parse_dates=['date'])
stores=pd.read_csv(r'originalni_datasetovi\stores.csv')
transactions=pd.read_csv(r'originalni_datasetovi\transactions.csv', parse_dates=['date'])

date_min_train=train.date.min()
date_max_train=train.date.max()
date_min_test=test.date.min()
date_max_test=test.date.max()

multi_index=pd.MultiIndex.from_product(
    [pd.date_range(date_min_train,date_max_train), train.store_nbr.unique(), train.family.unique()],
    names=['date','store_nbr','family'])
train=train.set_index(['date','store_nbr','family'])
train=train.reindex(multi_index)
train=train.reset_index()
train[['sales','onpromotion']]=train[['sales','onpromotion']].fillna(0)
train['id']=np.arange(0,train.shape[0])
train=train[['id']+[col for col in train.columns if col!='id']]

NO_FAMILIES=train.family.nunique()
NO_STORES=train.store_nbr.nunique()

transferred_holidays = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis = 1).reset_index(drop = True) #holidays that were transferred
transfer = holidays[(holidays.type == "Transfer")].drop("transferred", axis = 1).reset_index(drop = True) #days that they were transferred to (in ds they are always below the actual holiday so indexes are going to match)
tr = pd.concat([transferred_holidays,transfer], axis = 1) 
tr = tr.iloc[:, [5,1,2,3,4]]
holidays=holidays[(holidays.transferred==False) & (holidays.type !='Transfer')].drop('transferred',axis=1)
holidays=pd.concat([holidays,tr]).reset_index(drop=True)


def clean_description(desc):
    desc = re.sub(r'[+-]', '', desc)  # Remove + and -
    desc = re.sub(r'\d+', '', desc)  # Remove digits
    return desc

holidays["description"] = holidays["description"].apply(clean_description)
holidays["description"] = holidays["description"].str.replace("Puente ", "")
work_days=holidays[holidays['type']=='Work Day']
holidays=holidays[holidays['type']!='Work Day']
holidays.loc[holidays["description"].str.contains("futbol"), "description"] = "Futbol"

#Local holidays - city level
local_holidays=holidays[holidays['locale']=='Local']
local_holidays=local_holidays.rename(columns={'locale_name':'city','description': 'local_holidays'})
local_holidays=local_holidays.reset_index(drop=True)
local_holidays.drop(columns=['locale','type'],inplace=True)

local_holidays1=local_holidays.copy()
local_holidays2=local_holidays.copy()
local_holidays1['local_holidays'] = local_holidays1['local_holidays'].apply(lambda x: 'L ' + x)

#Regional holidays
regional_holidays=holidays[holidays['locale']=='Regional']
regional_holidays=regional_holidays.rename(columns={'locale_name':'state', 'description':'regional_holidays'})
regional_holidays=regional_holidays.reset_index(drop=True)
regional_holidays.drop(columns=['locale','type'],inplace=True)

regional_holidays1=regional_holidays.copy()
regional_holidays2=regional_holidays.copy()
regional_holidays1['regional_holidays'] = regional_holidays1['regional_holidays'].apply(lambda x: 'R ' + x)

#National holidays adn events
national_holidays=holidays[holidays['locale']=='National']
national_holidays=national_holidays.rename(columns={'description':'national_holidays'})
national_holidays=national_holidays.reset_index(drop=True)
national_holidays.drop(columns=['locale','type','locale_name'],inplace=True)

national_holidays1=national_holidays.copy()
national_holidays2=national_holidays.copy()
national_holidays1['national_holidays'] = national_holidays1['national_holidays'].apply(lambda x: 'N ' + x)

df = pd.concat([train, test]).reset_index(drop=True)
df['id']=np.arange(0,df.shape[0])
df=pd.merge(df,stores)

#work days
work_days=work_days[['date','type']].rename(columns={'type':'work_day'}).reset_index(drop=True)
work_days.work_day=work_days.work_day.notna().astype(int)
df=pd.merge(df,work_days, how='left', on='date')
df['work_day']=df['work_day'].fillna(0).astype(int)

df2=df.copy()

#####1
local_holidays_wide1 = local_holidays1.pivot_table(index='date', columns='local_holidays', aggfunc='size', fill_value=0)
local_holidays_wide1 = local_holidays_wide1.astype(int)
local_holidays_wide1 = local_holidays_wide1.reset_index() #to return date as a column
df=pd.merge(df,local_holidays_wide1, how='left', on=['date'])
for col in local_holidays_wide1.columns:
    if col != 'date':
        df[col] = df[col].fillna(0).astype(int)
regional_holidays_wide1 = regional_holidays1.pivot_table(index='date', columns='regional_holidays', aggfunc='size', fill_value=0)
regional_holidays_wide1 = regional_holidays_wide1.astype(int)
regional_holidays_wide1 = regional_holidays_wide1.reset_index() #to return date as a column
df=pd.merge(df,regional_holidays_wide1, how='left', on=['date'])
for col in regional_holidays_wide1.columns:
    if col != 'date':
        df[col] = df[col].fillna(0).astype(int)
national_holidays_wide1 = national_holidays1.pivot_table(index='date', columns='national_holidays', aggfunc='size', fill_value=0)
national_holidays_wide1 = national_holidays_wide1.astype(int)
national_holidays_wide1 = national_holidays_wide1.reset_index() #to return date as a column
df=pd.merge(df,national_holidays_wide1, how='left', on=['date'])
for col in national_holidays_wide1.columns:
    if col != 'date':
        df[col] = df[col].fillna(0).astype(int)
###############

#####2
local_holidays2.loc[local_holidays2["local_holidays"].str.contains("Fundacion"), "local_holidays"] = "Fundacion"
local_holidays2.loc[local_holidays2["local_holidays"].str.contains("Cantonizacion"), "local_holidays"] = "Cantonizacion"
local_holidays2.loc[local_holidays2["local_holidays"].str.contains("Independencia"), "local_holidays"] = "Independencia"
regional_holidays2.loc[regional_holidays2["regional_holidays"].str.contains("Provincializacion"), "regional_holidays"] = "Provincializacion"
local_holidays_wide2 = local_holidays2.pivot_table(index=['date','city'], columns=['local_holidays'], aggfunc='size', fill_value=0)
local_holidays_wide2 = local_holidays_wide2.astype(int)
local_holidays_wide2 = local_holidays_wide2.reset_index() #to return date as a column
df2=pd.merge(df2,local_holidays_wide2, how='left', on=['date', 'city'])
for col in local_holidays_wide2.columns:
    if col not in ['date', 'city']:
        df2[col] = df2[col].fillna(0).astype(int)
regional_holidays_wide2 = regional_holidays2.pivot_table(index=['date','state'], columns=['regional_holidays'], aggfunc='size', fill_value=0)
regional_holidays_wide2 = regional_holidays_wide2.astype(int)
regional_holidays_wide2 = regional_holidays_wide2.reset_index()
df2=pd.merge(df2,regional_holidays_wide2, how='left', on=['date','state'])
for col in regional_holidays_wide2.columns:
    if col not in ['date', 'state']:
        df2[col] = df2[col].fillna(0).astype(int)
national_holidays_wide2 = national_holidays2.pivot_table(index='date', columns='national_holidays', aggfunc='size', fill_value=0)
national_holidays_wide2 = national_holidays_wide2.astype(int)
national_holidays_wide2 = national_holidays_wide2.reset_index() #to return date as a column
df2=pd.merge(df2,national_holidays_wide2, how='left', on=['date'])
for col in national_holidays_wide2.columns:
    if col != 'date':
        df2[col] = df2[col].fillna(0).astype(int)
###############

oil.rename(columns={'dcoilwtico':'oil_price'},inplace=True)
oil = oil.merge(
    pd.DataFrame({"date": pd.date_range(date_min_train, date_max_test)}),
    on="date",
    how="outer",
).sort_values("date", ignore_index=True)
oil['oil_price'] = oil['oil_price'].interpolate(method='linear')
oil['oil_price'] = oil['oil_price'].fillna(method='bfill') #for the first missing value

df=pd.merge(df,oil,how='left',on='date')
df2=pd.merge(df2,oil,how='left',on='date')

missing_transactions_dates=pd.date_range(date_min_train, date_max_test).difference(transactions.date)
store_counts = transactions.groupby('date').size()
missing_stores_dates = store_counts[store_counts < 54].index.tolist()
missing_stores = []
for date in missing_stores_dates:
    existing_stores = transactions[transactions['date'] == date]['store_nbr'].tolist()
    missing_stores_for_date = list(set(range(1, 55)) - set(existing_stores))
    for store in missing_stores_for_date:
        missing_stores.append({"date": date, "store_nbr": store})
missing_stores_df = pd.DataFrame(missing_stores)
transactions = pd.concat([transactions, missing_stores_df], ignore_index=True).sort_values(["date", "store_nbr"])
transactions=transactions.reset_index(drop=True)
transactions=transactions.fillna(0)
date_range_df = pd.DataFrame({"date": pd.date_range(date_min_train, date_max_test)})
store_numbers_df = pd.DataFrame({
    "store_nbr": list(range(1, 55)) * len(date_range_df),
    "date": sorted(list(date_range_df['date']) * 54)
})
store_numbers_df=store_numbers_df[store_numbers_df['date'].isin(missing_transactions_dates)]
sum_of_sales=df[df['date'].isin(missing_transactions_dates)].groupby('date')['sales'].sum()
zero_transaction_dates=sum_of_sales[sum_of_sales==0].index.tolist()
for date in zero_transaction_dates:
    transactions.loc[transactions['date']==date, 'transactions']=0
transactions.transactions = transactions.groupby("store_nbr", group_keys=False).transactions.apply(
    lambda x: x.interpolate(method="linear", limit_direction="both")
)

df=pd.merge(df,transactions, how='left', on=['date','store_nbr'])
df2=pd.merge(df2,transactions, how='left', on=['date','store_nbr'])

df=df.sort_values(by=['date', 'store_nbr']).reset_index(drop=True)
df2=df2.sort_values(by=['date', 'store_nbr']).reset_index(drop=True)

df=df.drop(columns='id')
df2=df2.drop(columns='id')

df.to_csv('novi_datasetovi/train_test_v1.csv')
df2.to_csv('novi_datasetovi/train_test_v2.csv')