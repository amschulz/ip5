
# coding: utf-8

# In[4]:


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


# In[5]:


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

# get quality values for article
def get_quality(test, forecast, printPER=False):
    percantege_error = []
    test_without_0 = test[test != 0.0]
    mean = test.mean()
    default_for_0 =  mean / 10 if mean else 0.1
    for i,v in enumerate(test):
        percantege_error.append(math.sqrt(((forecast[i] / max(v, default_for_0)) - 1)**2))
    if printPER:
        print('PER: %s' % percantege_error)
    mape = sum(percantege_error) / float(len(percantege_error))
    median = statistics.median(percantege_error)
    return {'mape': mape,
            'median': median}

# get moving average with window w.
def get_ma(ser, w=2):
    ma = ser.rolling(window=w).mean()
    ma = ma.drop(0)
    ma.index = range(0, len(ma))
    return ma
    


# In[47]:


yes = [4,5,6,7,8, 9, 10]
ob_quartil = []
un_quartil = []
ob_whisker = []
un_whisker = []
median_abc =   []

# for every prediction with different years input rad in the file
for ye in yes:
    file = 'outputs/SARIMA_Pipeline_%s_Jahre_v3.csv' % ye
    df_read = pd.read_csv(filepath_or_buffer=file,
                     sep=';',
                     header=0,
                     index_col=0)
    df_hortima = pd.read_csv(filepath_or_buffer='outputs/Prognose_Hortima_2018.csv',
                             sep=';',
                             header=0,
                             index_col=0)
    df_read = df_read.drop(columns=['Mape', 'Median', 'TrainData Mean'])
    for i in range(1,13):
        df_read['Hortima M%s' % str(i)] = df_hortima[str(i)]
    
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
	
	# iterate over every article and get quality
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

        # turn lists into pd.series 
        series_test = pd.Series(test_data)
        series_forecast = pd.Series(forecast_data)
        series_hortima = pd.Series(hortima_data)

        # get quality for actual prediction
        quality = get_quality(series_test, series_forecast)
        mape.append(quality['mape'])

        quality_hortima = get_quality(series_test, series_hortima)
        mape_hortima.append(quality_hortima['mape'])

	# append quality values to df.
    df_read = df_read.drop(columns=drop_col)
    df_read['Mape Forecast'] = mape
    df_read['Mape Hortima'] = mape_hortima
	
	# exclude articles which have errors or were skipped
    print('Amount of articles skipped: %s' % len(df_read[df_read['Skipped'] == True]))
    print('Amount of articles with errors:  %s' % len(df_read[df_read['Error'] == True]))
    df_read = df_read[df_read['Skipped'] == False]
    df_read = df_read[df_read['Error'] == False]
    checkdf = df_read.isna()
	
	# display boxplots for specfific columns
    for col in df_read.columns:
        if ('Mape' in col) and (not 'Hortima' in col):
            an_col = df_read[col]         

            f, ax = plt.subplots(1, 3)
            f.set_size_inches(20, 4)
            ax[0].boxplot(an_col, showfliers=False)
            ax[0].set_title('Boxplot %s ohne Ausreisser %s Jahre' % (col, ye))
            B = ax[1].boxplot(an_col, showfliers=True)
            ax[1].set_title('Boxplot %s mit Ausreisser %s Jahre' % (col, ye))
            ob_quartil.append(B['boxes'][0].get_ydata()[2])
            un_quartil.append(B['boxes'][0].get_ydata()[0])
            ob_whisker.append(B['whiskers'][0].get_ydata()[1])
            un_whisker.append(B['whiskers'][1].get_ydata()[1])
            median_abc.append(B['medians'][0].get_ydata()[0])
            display((B['medians'][0].get_ydata()[0]))
            t = '\n'.join(
                ('Anzahl Werte: %s' % len(an_col),
                 'Anzahl Werte x > 1: %s' % len(an_col[an_col > 1]),
                 'Anzahl Werte 1 >= x > 0.5 : %s' % len(an_col[(an_col > 0.5) & (an_col <= 1)]),
                 'Anzahl Werte 0.5 >= x: %s' % len(an_col[an_col <= 0.5]),
                 #'Mittelwert: %s' % (an_col.mean())
                )
            )
            ax[2].text(0.05, 0.6,t,fontsize=16,)
            ax[2].get_xaxis().set_ticks([])
            ax[2].get_yaxis().set_ticks([])
            plt.show()

# display development of boxplot-values over different years
plt.title("Verlauf Boxplot-Werte")
plt.plot(pd.Series(data=ob_quartil, index=yes), color='red')
plt.plot(pd.Series(data=un_quartil, index=yes), color='blue')
plt.plot(pd.Series(data=ob_whisker, index=yes), color='green')
plt.plot(pd.Series(data=un_whisker, index=yes), color='black')
plt.plot(pd.Series(data=median_abc, index=yes), color='orange')
plt.legend(handles=[mpatches.Patch(color='red', label='Oberes Quartil'),
                    mpatches.Patch(color='blue', label='Unteres Quartil'),
                    mpatches.Patch(color='green', label='Unterer Whisker'),
                    mpatches.Patch(color='black', label='Oberer Whisker'),
                    mpatches.Patch(color='orange', label='Median')])
plt.show()

