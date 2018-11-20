
# coding: utf-8

# In[54]:


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


# In[55]:


default_startdate = date(2010, 1, 1)
default_enddate = date(2018, 10, 1)

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
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_161118/export_november_16.txt',
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
    default_for_0 = np.min(test_without_0) if len(test_without_0) > 5 else 0.1
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
    


# In[56]:


df_read = pd.read_csv(filepath_or_buffer='outputs/SARIMA_Pipeline_2018-11-17_03-01-09.csv',
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


# In[57]:


for i in range(1,13):
    df_read['Hortima M%s' % str(i)] = df_hortima[str(i)]
    
#display(df_read.head())


# In[58]:


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
    for i in range(1, 11):
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
df_read['MA Median Hortima'] = ma_median_hortima
df_read['MA Mape Forecast'] = ma_mape
df_read['MA Mape Hortima'] = ma_mape_hortima
df_read['Median Forecast'] = median
df_read['Median Hortima'] = median_hortima
df_read['Mape Forecast'] = mape
df_read['Mape Hortima'] = mape_hortima
df_read = df_read[df_read['Skipped'] == False]
df_read = df_read[df_read['Error'] == False]
display(df_read.head())


# In[59]:


checkdf = df_read.isna()
df_read = df_read[checkdf['MA Median Hortima'] == False]


# In[60]:


print('Analyse')
for col in df_read.columns:
    if 'Mape' in col:
        an_col = df_read[col]
        # an_col = an_col.dropna()
        t = '\n'.join(
            ('Anzahl Werte: %s' % len(an_col),
             'Anzahl Werte x > 1: %s' % len(an_col[an_col > 1]),
             'Anzahl Werte 1 >= x > 0.5 : %s' % len(an_col[(an_col > 0.5) & (an_col <= 1)]),
             'Anzahl Werte 0.5 >= x: %s' % len(an_col[an_col <= 0.5])
            )
        )
        f, ax = plt.subplots(1, 3)
        f.set_size_inches(20, 4)
        ax[0].boxplot(an_col, showfliers=False)
        ax[0].set_title('Boxplot %s ohne Ausreisser' % col)
        ax[1].boxplot(an_col, showfliers=True)
        ax[1].set_title('Boxplot %s mit Ausreisser' % col)
        ax[2].text(0.05, 0.6,t,fontsize=16,)
        ax[2].get_xaxis().set_ticks([])
        ax[2].get_yaxis().set_ticks([])

        plt.show()


# In[61]:


df_better_ma_mape = df_read[df_read['MA Mape Forecast'] < df_read['MA Mape Hortima']]
df_better_mape = df_read[df_read['Mape Forecast'] < df_read['Mape Hortima']]
display(df_better_ma_mape.head())
display(len(df_better_ma_mape))
display(df_better_ma_mape.head())
display(len(df_better_mape))
# display(len(df_read[(df_read['MA Mape Forecast'] <= df_read['MA Mape Hortima']) & df_read['Mape Forecast'] <= df_read['Mape Hortima']]))


# In[62]:


df_mape_below_05 = df_read[df_read['Mape Forecast'] < 0.5]
display(df_mape_below_05)


# In[63]:


start = date(2010, 1, 1)
end = date(2018, 1, 1)
def show_article_details(id='', showPrams=False, showPreviousDataPlot=False):
    if showPreviousDataPlot:
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
        
    df_read_article = pd.read_csv(filepath_or_buffer='outputs/SARIMA_Pipeline_2018-11-17_03-01-09.csv',
                     sep=';',
                     header=0,
                     index_col=0)
    df_hortima = pd.read_csv(filepath_or_buffer='outputs/Prognose_Hortima_2018.csv',
                         sep=';',
                         header=0,
                         index_col=0)
    article_df = df_read_article[df_read_article.index== id]
    # display(article_df)
    index = article_df.index[0]
    test_data = []
    forecast_data = []
    hortima_data = []
    for i in range(1, 11):
        test_data.append(article_df['Test M%s' % i][index])
        forecast_data.append(article_df['Forecast M%s' %i ][index])
        hortima_data.append(df_hortima[str(i)][index])
    #print('Test')
    #print(test_data)
    #print ('Forecast')
    #print (forecast_data)
    plt.title('Artikel %s im 2018' % id)
    plt.plot(test_data, color='blue')
    plt.plot(forecast_data, color='black')
    plt.plot(hortima_data, color='orange')
    plt.legend(handles=[
                    mpatches.Patch(color='blue', label='Test-Daten'),
                    mpatches.Patch(color='black', label='Forecast'),
                    mpatches.Patch(color='orange', label='Hortima Forecast'),

                   ])
    plt.show()
    if showPrams:
        print('Known Params: %s' % article_df['Known Params'][index])
        print('p: %s' % article_df['p'][index])
        print('d: %s' % article_df['d'][index])
        print('q: %s' % article_df['q'][index])
        print('P: %s' % article_df['P'][index])
        print('D: %s' % article_df['D'][index])
        print('Q: %s' % article_df['Q'][index])
        print('S: %s' % article_df['S'][index])

show_article_details(4592, True, True)


# In[64]:


for index, row in df_mape_below_05.iterrows(): 
    show_article_details(index, False, False)

