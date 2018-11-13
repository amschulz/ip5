
# coding: utf-8

# In[2]:


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


# In[3]:


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


# In[4]:


df_read = pd.read_csv(filepath_or_buffer='outputs/SARIMA_Pipeline_2018-11-11_00-58-37.csv',
                 sep=';',
                 header=0)
#df_read = results_df
#df_read['ArtikelID'] = df_read.index
display(df_read)
# display(df[(df['RMSE / TrainData Mean'] > 0.3) & (df['RMSE / TrainData Mean'] < 2)])


# In[5]:


print('Analyse')
boxplot_median = df_read['Median'].dropna()
boxplot_mape = df_read['Mape'].dropna()

plt.title('Box-Plot des Medians')
plt.boxplot(boxplot_median, showfliers=False)
plt.show()
plt.title('Box-Plot des Mapes')
plt.boxplot(boxplot_mape, showfliers=False)
plt.show()

print('Mittelwert des Medians: %s' % boxplot_median.mean())
print('Mittelwert des Mapes: %s' % boxplot_mape.mean())
print('Median des Medians: %s' % statistics.median(boxplot_median))
print('Median des Mapes: %s' % statistics.median(boxplot_mape))
print('Anzahl übersprungener Artikel: %s' % len(df_read[df_read['Skipped']== True]))
print('Anzahl Artikel mit Errors: %s' % len(df_read[df_read['Error'] == True]))
print('Anzahl Artikel mit MAPE über 1.0: %s' % len(boxplot_mape[boxplot_mape > 1.0]))
print('Anzahl Artikel mit MAPE kleiner oder gleich 1.0: %s' % len(boxplot_mape[boxplot_mape <= 1.0]))
print('Anzahl Artikel mit MAPE über 0.5: %s' % len(boxplot_mape[boxplot_mape > 0.5]))
print('Anzahl Artikel mit MAPE kleiner oder gleich 0.5: %s' % len(boxplot_mape[boxplot_mape <= 0.5]))
print('Anzahl Artikel mit Median über 1.0: %s' % len(boxplot_median[boxplot_median > 1.0]))
print('Anzahl Artikel mit Median kleiner oder gleich 1.0: %s' % len(boxplot_median[boxplot_median <= 1.0]))
print('Anzahl Artikel mit Median über 0.5: %s' % len(boxplot_median[boxplot_median > 0.5]))
print('Anzahl Artikel mit Median kleiner oder gleich 0.5: %s' % len(boxplot_median[boxplot_median <= 0.5]))
# display(df_read[df_read['Mape'] <= 0.5])


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


# In[3]:


df1 = pd.read_csv(filepath_or_buffer='outputs/SARIMA_Pipeline_2018-11-11_00-58-37.csv',
                 sep=';',
                 header=0)
artIds = df1['ArtikelID'].unique()

ma_median = []
ma_mape = []

for index, artId in enumerate(artIds):
    test_data = []
    forecast_data = []
    for i in range(1, 13):
        test_data.append(df1['Test M%s' % i][index])
        forecast_data.append(df1['Forecast M%s' %i ][index])
    series_test = pd.Series(test_data)
    series_forecast = pd.Series(forecast_data)
    dfdata = {'mape': df1['Mape'][index], 'median': df1['Median'][index]}
    test_ma2 = series_test.rolling(window=2).mean()
    test_ma2 = test_ma2.drop(0)
    test_ma2.index = range(0, len(test_ma2))
    forecast_ma2 = series_forecast.rolling(window=2).mean()
    forecast_ma2 = forecast_ma2.drop(0)
    forecast_ma2.index = range(0, len(forecast_ma2))
    #plt.plot(test_data)
    #plt.plot(forecast_data)
    #plt.plot(test_ma2)
    #plt.plot(forecast_ma2)
    #plt.show()
    ma_quality = get_quality(test_ma2, forecast_ma2)
    ma_median.append(ma_quality['median'])
    ma_mape.append(ma_quality['mape'])

df1['MA Median'] = ma_median
df1['MA Mape'] = ma_mape

print(df1['Median'].dropna().mean())
print(df1['MA Median'].dropna().mean())
print(df1['Mape'].dropna().mean())
print(df1['MA Mape'].dropna().mean())


# In[16]:


plt.title('Box-Plot des MA Medians')
plt.boxplot(df1['MA Median'].dropna(), showfliers=False)
plt.show()
plt.title('Box-Plot des MA Mapes')
plt.boxplot(df1['MA Mape'].dropna(), showfliers=False)
plt.show()


# In[4]:


plt.title('Box-Plot des MA Medians')
plt.boxplot(df1['MA Median'].dropna(), showfliers=True)
plt.show()
plt.title('Box-Plot des MA Mapes')
plt.boxplot(df1['MA Mape'].dropna(), showfliers=True)
plt.show()


# In[6]:


display(df1[df1['MA Median'] > 10000])
display(df1[df1['MA Mape'] > 10000])

