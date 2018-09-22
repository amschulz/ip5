
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import date
from statsmodels.tsa.stattools import adfuller
import math


# In[80]:


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
    df = df.reindex(idx, fill_value=0)
    return df

def get_dataframe(ids_to_track=[]):
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_200818/export_august.txt',
                     sep=';',
                     header=0,
                     usecols=[0,6,8,10])
    df = df.rename(index=str, columns={'fk_ArtikelBasis_ID_l':'ArtikelID'})

    if ids_to_track:
        df = df[(df['ArtikelID'].isin(ids_to_track))]

     # convert FaktDatum to datetime
    df['Datum'] = pd.to_datetime(df['FaktDatum'], dayfirst=True, errors='raise')
    return df

def getStationaryData(y_train):
    drange = range(0,3)
    y_train_stationary = y_train
    for d in drange:
        dftest = adfuller(y, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value ({})'.format(key)] = value
        adf_val = dftest[0]

        if adf_val < dftest[4]['5%']:
            return (y_train_stationary, d)
        else:
            y_train_stationary = np.diff(y_train_stationary)
    print('Never stationary!')
    return (y_train, 0)

def get_best_model(y_train):
    p = 0
    d = 0
    q = 0
    P = 2
    D = 0
    Q = 0
    S = 12
    y_train_stationary, d = getStationaryData(y_train)
    
    acf = sm.tsa.stattools.acf(y_train_stationary, nlags=36, alpha=0.1)
    pacf = sm.tsa.stattools.pacf(y_train_stationary, nlags=36, alpha=0.1)
    acf_value = acf[0]
    acf_limit_below = [(x[0] - acf_value[i]) for i, x in enumerate(acf[1].tolist())]
    acf_limit_above = [(x[1] - acf_value[i]) for i, x in enumerate(acf[1].tolist())]
    pacf_value = pacf[0]
    pacf_limit_below =[(x[0] - pacf_value[i]) for i, x in enumerate(pacf[1].tolist())]
    pacf_limit_above = [(x[1] - pacf_value[i]) for i, x in enumerate(pacf[1].tolist())]
    acf_peaks = [i for i, x in enumerate(acf_value) if (x < acf_limit_below[i] or x > acf_limit_above[i])]
    pacf_peaks = [i for i, x in enumerate(pacf_value) if (x < pacf_limit_below[i] or x > pacf_limit_above[i])]
    acf_peaks = acf_peaks[1:]
    pacf_peaks = pacf_peaks[1:]
    
    param = (p, d, q)
    param_seasonal = (P, D, Q, S)
    return sm.tsa.statespace.SARIMAX(y_train,
                                     order = param,
                                     seasonal_order = param_seasonal,
                                     enforce_stationarity=True,
                                     enforce_invertibility=True)

def get_rmse(test, forecast):
    mse = ((test - forecast) ** 2).mean()
    return math.sqrt(mse)


# In[81]:


ids = [722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830]
start = date(2010, 1, 1)
end = date(2018, 1, 1)
df = get_dataframe(ids_to_track=ids)


# In[86]:


rmses = []
articleIds = df['ArtikelID'].unique()
for articleId in articleIds:
    dfTemporary = df[(df['ArtikelID'].isin([articleId]))]
    dfTemporary = dfTemporary.drop(columns=['ArtikelID'])
    dfTemporary = group_by_frequence(dfTemporary, frequence='MS',startdate=start, enddate=end)
    dfTemporary = dfTemporary.drop(columns=['Datum'])
    y = dfTemporary['Menge']
    y_train = y[:date(2016,12,31)]
    X_train = y_train.index
    y_test = y[date(2017,1,1):]
    X_test = y_test.index
    model = get_best_model(y_train)
    fitted_model = model.fit()
    pred = fitted_model.get_prediction(dynamic=False,
                                       steps=12,
                                       start=pd.to_datetime('2016-12-01'),
                                       end=pd.to_datetime('2017-12-31'))
    plt.figure(figsize=(20,5))
    plt.plot(pred2, color='red')
    plt.plot(y_test, color='blue')
    plt.show()
    rmses.append(get_rmse(y_test, pred.predicted_mean) / y_train.mean())
    break
    
summe = 0
for rmse in rmses:
    summe += rmse
display('RMSE Mean: %s' % (summe/len(rmses)))


# In[83]:


articleIds = df['ArtikelID'].unique()
for articleId in articleIds:
    dfTemporary = df[(df['ArtikelID'].isin([articleId]))]
    dfTemporary = dfTemporary.drop(columns=['ArtikelID'])
    dfTemporary = group_by_frequence(dfTemporary, frequence='MS',startdate=start, enddate=end)
    dfTemporary = dfTemporary.drop(columns=['Datum'])
    y = dfTemporary['Menge']
    y_train = y[:date(2016,12,31)]
    X_train = y_train.index
    y_test = y[date(2017,1,1):]
    X_test = y_test.index
    model = get_best_model(y_train)
    fitted_model = model.fit()
    pred = fitted_model.get_prediction(dynamic=False,
                                       steps=12,
                                       start=pd.to_datetime('2016-12-01'),
                                       end=pd.to_datetime('2017-12-31'))
    pred2 = fitted_model.forecast(dynamic=True, steps=12)
    plt.figure(figsize=(20,5))
    plt.plot(y[date(2016,1,1):date(2016,12,31)], color='orange')
    plt.plot(y[date(2015,1,1):date(2015,12,31)], color='orange')
    plt.plot(pred2, color='red')
    plt.plot(y_test, color='blue')
    # plt.plot(pred.predicted_mean, color='green')
    plt.show()
    display(y[date(2015,11,1):date(2015,12,31)])
    display(y[date(2016,11,1):date(2016,12,31)])
    display(pred2[date(2017,11,1):date(2017,12,31)])
    display(pred.predicted_mean[date(2017,11,1):date(2017,12,31)])
    break

