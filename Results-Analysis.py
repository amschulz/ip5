
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import math
import itertools
import matplotlib.patches as mpatches
import statistics

from datetime import date
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose


# In[98]:


default_startdate = date(2010, 1, 1)
default_enddate = date(2018, 1, 1)

def group_by_frequence(df,
                       frequence='MS',
                       startdate=default_startdate,
                       enddate=default_enddate):
    df = df.copy()
    df = df.groupby([pd.Grouper(key='Datum', freq=frequence)])['Menge'].sum().reset_index()
    idx = pd.date_range(start=startdate, end=enddate, freq=frequence)
    df.index = pd.DatetimeIndex(df.Datum)
    
    # fill in 0 if there are no values for a frequence-element
    df = df.reindex(idx, fill_value=0)
    return df

def get_dataframe(ids_to_track=[]):
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_200818/export_august.txt',
                     sep=';',
                     header=0,
                     usecols=[0,6,8,10])
    # rename column
    df = df.rename(index=str, columns={'fk_ArtikelBasis_ID_l':'ArtikelID'})

    # filter for ids which are given as param
    if ids_to_track:
        df = df[(df['ArtikelID'].isin(ids_to_track))]

     # convert FaktDatum to datetime
    df['Datum'] = pd.to_datetime(df['FaktDatum'], dayfirst=True, errors='raise')
    return df




def get_quality(test, forecast):
    percantege_error = []
    test_without_0 = test[test != 0.0]
    default_for_0 = np.min(test_without_0) if len(test_without_0) > 6 else 1
    for i,v in enumerate(test):
        # print('Test-Value %s - Forecast-Value %s' % (v, forecast[i]))
        percantege_error.append(math.sqrt(((forecast[i] / max(v, default_for_0)) - 1)**2))
    mape = sum(percantege_error) / float(len(percantege_error))
    median = statistics.median(percantege_error)
    # mse = ((test - forecast) ** 2).mean()
    # original_rmse = math.sqrt(mse)
    # print('alternative_rmse: %s - alternativ_rmse2: %s original_rmse: %s' %  (alternative_rmse, alternative_rmse2, original_rmse))
    return {'mape': mape,
            'median': median}

def get_ma(ser, w=2):
    ma = ser.rolling(window=w).mean()
    ma = ma.drop(0)
    ma.index = range(0, len(ma))
    return ma
    


# In[99]:


df_read = pd.read_csv(filepath_or_buffer='outputs/SARIMA_Pipeline_2018-11-11_00-58-37.csv',
                 sep=';',
                 header=0,
                 index_col=0)
df_hortima = pd.read_csv(filepath_or_buffer='outputs/Prognose_Hortima_2018.csv',
                         sep=';',
                         header=0,
                         index_col=0)
df_read = df_read.drop(columns=['Mape', 'Median', 'TrainData Mean'])
#display(df_read.head())
#display(df_hortima.loc[[1744, 1357, 1676, 2355, 4810]])


# In[100]:


for i in range(1,13):
    df_read['Hortima M%s' % str(i)] = df_hortima[str(i)]
    
# display(df_read.head())


# In[101]:


artIds = df_read.index

median = []
mape = []
median_hortima = []
mape_hortima = []
ma_median = []
ma_mape = []
ma_median_hortima = []
ma_mape_hortima = []
drop_col = []

for ind, artId in enumerate(artIds):
    test_data = []
    forecast_data = []
    hortima_data= []
    for i in range(1, 13):
        test_data.append(df_read['Test M%s' % i][artId])
        forecast_data.append(df_read['Forecast M%s' % i][artId])
        hortima_data.append(df_read['Hortima M%s' % i][artId])
        if ind == 0:
            drop_col.append('Test M%s' % i)
            drop_col.append('Forecast M%s' % i)
            drop_col.append('Hortima M%s' % i)
        
    #if artId == 1744:
    #    plt.plot(test_data, color='red')
    #    plt.plot(forecast_data, color='blue')
    #    plt.plot(hortima_data, color='orange')
    #    plt.show()
    
    # turn lists into pd.series 
    series_test = pd.Series(test_data)
    series_forecast = pd.Series(forecast_data)
    series_hortima = pd.Series(hortima_data)
    
    # get quality for actual prediction
    quality = get_quality(series_test, series_forecast)
    median.append(quality['median'])
    mape.append(quality['mape'])
    
    quality_hortima = get_quality(series_test, series_hortima)
    median_hortima.append(quality_hortima['median'])
    mape_hortima.append(quality_hortima['mape'])
    
    # get quality for ma of prediction
    test_ma2 = get_ma(series_test, 2)
    forecast_ma2 = get_ma(series_forecast, 2)
    hortima_ma2 = get_ma(series_hortima, 2)
    
    ma_quality = get_quality(test_ma2, forecast_ma2)
    ma_median.append(ma_quality['median'])
    ma_mape.append(ma_quality['mape'])
    
    ma_quality_hortima = get_quality(test_ma2, hortima_ma2)
    ma_median_hortima.append(ma_quality_hortima['median'])
    ma_mape_hortima.append(ma_quality_hortima['mape'])

df_read = df_read.drop(columns=drop_col)
df_read['MA Median Forecast'] = ma_median
df_read['MA Mape Forecast'] = ma_mape
df_read['MA Median Hortima'] = ma_median_hortima
df_read['MA Mape Hortima'] = ma_mape_hortima
df_read['Median Forecast'] = median
df_read['Mape Forecast'] = mape
df_read['Median Hortima'] = median_hortima
df_read['Mape Hortima'] = mape_hortima
df_read[df_read['Skipped']== True]
df_read[df_read['Error']== True]
display(df_read.head())


# In[102]:


print('Analyse')
for col in df_read.columns:
    if 'Median' in col or 'Mape' in col:
        an_col = df_read[col]
        an_col.dropna()
        plt.title(col)
        plt.boxplot(boxplot_median, showfliers=False)
        plt.boxplot(boxplot_median, showfliers=True)
        plt.show()


# In[6]:


start = date(2010, 1, 1)
end = date(2017, 1, 1)
def show_article_details(id=''):
    orig_df = get_dataframe([id])
    dfTemporary = orig_df[(orig_df['ArtikelID'].isin([id]))]
    dfTemporary = dfTemporary.drop(columns=['ArtikelID'])
    dfTemporary = group_by_frequence(dfTemporary, frequence='MS',startdate=start, enddate=end)
    dfTemporary = dfTemporary.drop(columns=['Datum'])
    # display(dfTemporary)
    plt.figure(figsize=(20,5))
    plt.plot(dfTemporary['Menge'])
    
    plt.title('Verkaufszahlen Artikel %s von %s bis %s' % (id, start, end))
    plt.show()
    
    article_df = df_read[df_read['ArtikelID']== id]
    # display(article_df)
    index = article_df.index[0]
    test_data = []
    forecast_data = []
    for i in range(1, 13):
        test_data.append(article_df['Test M%s' % i][index])
        forecast_data.append(article_df['Forecast M%s' %i ][index])
    #print('Test')
    #print(test_data)
    #print ('Forecast')
    #print (forecast_data)
    plt.title('Artikel %s im 2017' % id)
    plt.plot(test_data, color='blue')
    plt.plot(forecast_data, color='black')
    plt.legend(handles=[
                    mpatches.Patch(color='blue', label='Test-Daten'),
                    mpatches.Patch(color='black', label='Forecast'),
                   ])
    plt.show()
    print('Mape: %s' % article_df['Mape'][index])
    print('Median: %s' % article_df['Median'][index])
    print('Known Params: %s' % article_df['Known Params'][index])
    print('p: %s' % article_df['p'][index])
    print('d: %s' % article_df['d'][index])
    print('q: %s' % article_df['q'][index])
    print('P: %s' % article_df['P'][index])
    print('D: %s' % article_df['D'][index])
    print('Q: %s' % article_df['Q'][index])
    print('S: %s' % article_df['S'][index])

show_article_details(6673)

