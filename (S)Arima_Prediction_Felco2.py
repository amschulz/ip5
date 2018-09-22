
# coding: utf-8

# In[1]:


# %matplotlib notebook
# load required modules
import pandas as pd
from pandas import DataFrame

import numpy as np
import matplotlib.pyplot as plt
import itertools
import re
import os
import sys
import requests
import math

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

import scipy.stats as scs
from scipy.stats import norm

import datetime as dt
from datetime import datetime
from datetime import date

from io import BytesIO
from sklearn.linear_model import LinearRegression


# In[2]:


def group_by_frequence(df, frequence='W'):
    df = df.copy()
    df = df.groupby([pd.Grouper(key='Datum', freq=frequence)])['Menge'].sum().reset_index()
    idx = pd.date_range(start=startdate, end=enddate, freq=frequence)
    df.index = pd.DatetimeIndex(df.Datum)
    df = df.reindex(idx, fill_value=0)
    return df

def get_dataframe_SingleArt(filename, frequence='W'):
    df = pd.read_csv(filepath_or_buffer='../Datenexporte/Single/'+filename+'.csv',
                     sep=';',
                     header=0,
                     usecols=[0,1])
    
    df['Datum'] = pd.to_datetime(df['Datum'], yearfirst=True, errors='raise')

     # convert FaktDatum to datetime
    # df.index = pd.DatetimeIndex(df.Datum, yearfirst=True)
    return group_by_frequence(df, frequence)

# perform Augmented Dickey Fuller test
def adf_test(y):
    print('Results of Augmented Dickey-Fuller test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print(dfoutput)
    adf_val = dftest[0]

    if adf_val < dftest[4]['5%']:
        print('Is stationary')
        d = 0
    else:
        print('Is not stationary. Try differencing.')
        d = 1
        # TODO
    return d
        
def calculate_pdq(y, S):
    # define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 3)

    # generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], S) for x in list(itertools.product(p, d, q))]
    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    tmp_model = None
    best_mdl = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                tmp_mdl = sm.tsa.statespace.SARIMAX(y,
                                                    order = param,
                                                    seasonal_order = param_seasonal,
                                                    enforce_stationarity=True,
                                                    enforce_invertibility=True)
                res = tmp_mdl.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_mdl = tmp_mdl
            except:
                # print("Unexpected error:", sys.exc_info()[0])
                continue
    print("Best SARIMAX{}x{} model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))


# In[3]:


enddate = date(2018, 1, 1)
startdate = date(2010, 1, 1)

# load data from file
df = get_dataframe_SingleArt('felco2','M')

# prepare data
df = df.drop(columns=['Datum'])
y = df['Menge']

# make train and test set
splitdate = date(2016, 12, 31)
y_train = y[:splitdate]
y_test = y[splitdate:]
y_test = y_test[1:]
X_test = y_test.index.map(dt.datetime.toordinal).values.reshape(-1, 1)


# In[4]:


# Testing: what are the length of the different series
# sarimax needs conmplete series, to prevent errors in the calc of pdq

print ('original len: ', len(y_train))
print ('differenced:  ', len(np.diff(y_train)))


# In[5]:


# show training data as graph
plt.figure(figsize=(20,5))
plt.plot(y_train.index, y_train)

X_train = y_train.index.map(dt.datetime.toordinal).values.reshape(-1, 1)

# add regression line to chart
regr = LinearRegression()
regr.fit(X_train, y_train)
prediction = regr.predict(X_train)
plt.plot(X_train, prediction)

# make chart nice, fixed to 2010 to 2017
years = [startdate, date(2011, 1, 1), 
         date(2012, 1, 1),  date(2013, 1, 1), 
         date(2014, 1, 1), date(2015, 1, 1), 
         date(2016, 1, 1), date(2017, 1, 1)]
for year in years:
    plt.axvline(x=year, color='black')
plt.show()

# display(df.head())

# Hier sieht man eine Übersicht über die verwertenden Daten. 
# Die schwarzen Linien bezeichnen den Jahreswechsel und die 
# orange eine lineare Regression über die Daten.


# In[34]:


import statsmodels.tsa.api as smt
title=''
lags=24

y = y_train
# 1x diff
y_diff= np.diff(y_train)

# 2x diff
y_diff= np.diff(y_diff)

    # weekly moving averages (5 day window because of workdays)
rolling_mean = pd.rolling_mean(y, window=12)
rolling_std = pd.rolling_std(y, window=12)

fig = plt.figure(figsize=(14, 12))
layout = (4, 2)  # rows, cols
ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
acf_ax = plt.subplot2grid(layout, (1, 0))
pacf_ax = plt.subplot2grid(layout, (1, 1))
qq_ax = plt.subplot2grid(layout, (2, 0))
hist_ax = plt.subplot2grid(layout, (2, 1))
acfd_ax = plt.subplot2grid(layout, (3, 0))
pacfd_ax = plt.subplot2grid(layout, (3, 1))
    
# time series plot
y.plot(ax=ts_ax)
rolling_mean.plot(ax=ts_ax, color='crimson');
rolling_std.plot(ax=ts_ax, color='darkslateblue');
plt.legend(loc='best')
ts_ax.set_title(title, fontsize=24);
    
# acf and pacf
acfd = sm.tsa.stattools.acf(y, nlags=lags)
smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5) 
    
# qq plot
sm.qqplot(y, line='s', ax=qq_ax)
qq_ax.set_title('QQ Plot')
    
# hist plot
y.plot(ax=hist_ax, kind='hist', bins=25);
hist_ax.set_title('Histogram');

# differenced acf/pacf

y_diff= np.diff(y_train)

smt.graphics.plot_acf(y_diff, lags=lags, ax=acfd_ax, alpha=0.5)
smt.graphics.plot_pacf(y_diff, lags=lags, ax=pacfd_ax, alpha=0.5) 
acfd_ax.set_title('Differenced ACF')
pacfd_ax.set_title('Differenced PACF')
    
plt.tight_layout();
#plt.savefig('./img/{}.png'.format(filename))
plt.show()


# In[35]:



# optimal d is found at lowest std deviation - but NOT ALWAYS
print('std of original series      ', y_train.std())
print('std of differenced data     ', y_diff.std())
print('std of 2x differenced data  ', np.diff(y_diff).std())

print('------------------------------------------\n')

# adf test for stationarity. If y_train is stationary, d = 0
d = adf_test(y_train)
if (d > 0):
    print('------------------------------------------\n')
    d = adf_test(y_diff)
    if (d>0) :
        print('------------------------------------------\n')
        adf_test(np.diff(y_diff))


# In[ ]:


calculate_pdq(y_train, 12)


# In[ ]:


# results for felco

# weekly
#(0,0,1)(1,0,0,52) AIC  3048.007
#(0,0,1)(2,1,1,52) AIC  2570.698 ## << best

# monthly
#Best SARIMAX(0, 1, 1)x(1, 2, 1, 12) model - AIC:618.6424779676563

# eval by charts, monthly
# ARIMA (p d q)
p = 0 # AR part       rule 6: PACF lag 1 is negative. No AR
d = 1 # Differencing  adf says stationary, BUT acf drop not after 1
q = 1 # MA part       rule 7: ACF of differenced data lag-1 is negative

# Seasonal (P, D, Q, S)
P = 1 #             rule 13: lag 52 is positive, add SAR
D = 1 # diff        rule 12: series has seasonal pattern
Q = 0 #             rule 13 ...
S = 12              # 12 month per year


# In[36]:


p = 0
d = 1
q = 1

P = 2
D = 1
Q = 1

S = 12

param = (p, d, q)
param_seasonal = (P, D, Q, S)

model = sm.tsa.statespace.SARIMAX(y_train,
                                  order = param,
                                  seasonal_order = param_seasonal,
                                  enforce_stationarity=True,
                                  enforce_invertibility=True)
res = model.fit(disp=False)
print(res.summary())

res.plot_diagnostics(figsize=(16, 10))
plt.tight_layout()
plt.show()


# In[ ]:


# ARIMA prediction, not very useful so far
param = (10,0,0)
arima = ARIMA(y_train.astype('float32'), param, 
              exog=None, dates=None, 
              freq=None, missing='none')
res_arima = arima.fit()
print(res_arima.summary())

pdate_start = pd.to_datetime('2016-12-31')
pdate_end   = pd.to_datetime('2017-12-31')
chart_start = pd.to_datetime('2010-12-31')

fig, ax = plt.subplots()
yy = df['Menge']
ax = yy[chart_start:].plot(ax=ax)
fig = res_arima.plot_predict(pdate_start, pdate_end, dynamic=True, ax=ax, plot_insample=False)
plt.show()


# In[45]:


# SARIMA prediction

# in-sample-prediction and confidence bounds
# prediction time range
pdate_start = pd.to_datetime('2016-12-31')
pdate_end   = pd.to_datetime('2017-12-31')
chart_start = pd.to_datetime('2010-12-31')

pred = res.get_prediction(start=pdate_start, 
                          end=pdate_end,
                          dynamic=True)
pred_ci = pred.conf_int()
 
# plot in-sample-prediction

plt.figure(figsize=(20,10))
yy = df['Menge']
ax = yy[chart_start:].plot(label='Observed',color='#006699');
pred.predicted_mean.plot(ax=ax, label='One-step Ahead Prediction', alpha=.7, color='#ff0066');
 
# draw confidence bound (gray)
ax.fill_between(pred_ci.index, 
                pred_ci.iloc[:, 0], 
                pred_ci.iloc[:, 1], color='#ff0066', alpha=.25);
 
# style the plot
ax.fill_betweenx(ax.get_ylim(), pdate_start, y.index[-1], alpha=.15, zorder=-1, color='grey');
ax.set_xlabel('Date')
ax.set_ylabel('Sales')

plt.legend(loc='upper left')
plt.show()

y_hat = pred.predicted_mean
y_true = y[pdate_start:]
 
# compute the mean square error
mse = ((y_hat - y_true) ** 2).mean()
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))


# In[125]:



forecast = model_fitted.forecast(len(X_test))

# compute the mean square error
y_hat = forecast
y_true = y_test
mse = ((y_hat - y_true) ** 2).mean()

print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))

lastyear = y[date(2016,1,1):date(2016,12,31)]
plt.figure(figsize=(20,5))
plt.plot(y_test, color='blue')
plt.plot(forecast, color='red')
plt.show()

plt.figure(figsize=(20,5))
plt.plot(y.index, y)

X_train = y_train.index.map(dt.datetime.toordinal).values.reshape(-1, 1)
y_train = y_train
regr = LinearRegression()
regr.fit(X_train, y_train)
prediction = regr.predict(X_train)
plt.plot(X_train, prediction)


years = [startdate, date(2015, 1, 1), date(2016, 1, 1), date(2017, 1, 1)]
for year in years:
    plt.axvline(x=year, color='black')
    
plt.plot(forecast, color='red')
plt.show()

display(model_fitted.summary())


# In[ ]:


import itertools
import sys
import time
start = time.time()
# define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 4)
# generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
best_rmse = 10000000000000000000
best_aic = 10000000000000000000
best_seasonal_pdq = (-1,-1,-1)
best_seasonal_pdq_rmse = (-1,-1,-1)
best_shift = -5
shifts = [-2, -1, 0, 1, 2]
for param_seasonal in seasonal_pdq:
        try:
            tmp_mdl = sm.tsa.statespace.SARIMAX(y_train,
                                                order = (0,0,0),
                                                seasonal_order = param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=True)
            res = tmp_mdl.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_seasonal_pdq = param_seasonal
            forecast = res.forecast(len(X_test))
            y_hat = forecast
            y_true = y_test
            for shift in shifts:
                mse = ((y_hat.shift(shift) - y_true) ** 2).mean()
                rmse = math.sqrt(mse)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_seasonal_pdq_rmse = param_seasonal
                    best_shift = shift
        except Exception as e:
            print("Unexpected error: ", e)
            continue
end = time.time()
print('Duration: %s' % (end-start))
print("Best AIC SARIMAX{}x{} model - AIC:{}".format((0,0,0), best_seasonal_pdq, best_aic))
# Best SARIMAX(0, 0, 0)x(3, 3, 1, 52)12 model - AIC:4396.615156860976
print("Best RMSE SARIMAX{}x{} model - RMSE:{}".format((0,0,0), best_seasonal_pdq_rmse, best_rmse))


# Bemerkungen: 
#     1. Mit 3 Jahren Daten kann man nicht wirklich viel anfangen. Die meisten Seasonal-Parameter funktionieren nicht, da nicht genügend Daten vorhanden sind.
#     2. Für die ACF und PACF gilt die Werte bei Lag 0 zu ignorieren, da diese sowieso immer 1 sein sollten


# In[ ]:


param = (p, d, q)
param_seasonal = (P, D, Q, S)

model = sm.tsa.statespace.SARIMAX(y_train,
                                  order = param,
                                  seasonal_order = param_seasonal,
                                  enforce_stationarity=True,
                                  enforce_invertibility=True)
model_fitted = model.fit(disp=False)
forecast = model_fitted.forecast(len(X_test))

# compute the mean square error
y_hat = forecast
y_true = y_test
mse = ((y_hat - y_true) ** 2).mean()

print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))

lastyear = y[date(2016,1,1):date(2016,12,31)]
plt.figure(figsize=(20,5))
plt.plot(y_test, color='blue')
plt.plot(forecast, color='red')
plt.show()

plt.figure(figsize=(20,5))
plt.plot(y.index, y)

X_train = y_train.index.map(dt.datetime.toordinal).values.reshape(-1, 1)
y_train = y_train
regr = LinearRegression()
regr.fit(X_train, y_train)
prediction = regr.predict(X_train)
plt.plot(X_train, prediction)


years = [startdate, date(2015, 1, 1), date(2016, 1, 1), date(2017, 1, 1)]
for year in years:
    plt.axvline(x=year, color='black')
    
plt.plot(forecast, color='red')
plt.show()

display(model_fitted.summary())


# In[ ]:



enddate = date(2018, 1, 1)
default_startdate = date(2010, 1, 1)

def group_by_frequence(df, frequence='W', startdate=default_startdate):
    df = df.copy()
    df = df.groupby([pd.Grouper(key='Datum', freq=frequence)])['Menge'].sum().reset_index()
    idx = pd.date_range(start=startdate, end=enddate, freq=frequence)
    df.index = pd.DatetimeIndex(df.Datum)
    df = df.reindex(idx, fill_value=0)
    return df

def get_dataframe_SingleArt(filename, frequence='W', startdate=default_startdate):
    df = pd.read_csv(filepath_or_buffer='../Datenexporte/Single/'+filename+'.csv',
                     sep=';',
                     header=0,
                     usecols=[0,1])
    
    df['Datum'] = pd.to_datetime(df['Datum'], yearfirst=True, errors='raise')

     # convert FaktDatum to datetime
    # df.index = pd.DatetimeIndex(df.Datum, yearfirst=True)
    return group_by_frequence(df, frequence)

def test_params(frequency='MS', maxrange=2, article='bambus1201012', startdate=default_startdate):
    df = get_dataframe_SingleArt(article, frequency, startdate)
    df = df.drop(columns=['Datum'])
    y = df['Menge']
    if frequency == 'MS':
        splitdate = date(2016, 12, 1)
        s = 12
    elif frequency == 'W':
        splitdate = date(2016, 12, 25)
        s = 52
    y_train = y[:splitdate]
    y_test = y[splitdate:]
    y_test = y_test[1:]
    X_test = y_test.index.map(dt.datetime.toordinal).values.reshape(-1, 1)

    # plt.figure(figsize=(20,5))
    # plt.plot(y_train.index, y_train)

    X_train = y_train.index.map(dt.datetime.toordinal).values.reshape(-1, 1)
    y_train = y_train
    # regr = LinearRegression()
    # regr.fit(X_train, y_train)
    # prediction = regr.predict(X_train)
    # plt.plot(X_train, prediction)

    
    import itertools
    import sys
    import time
    start = time.time()
    # define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, maxrange)
    # generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, d, q))]
    
    best_seasonal_pdq_aic = []
    best_seasonal_pdq_rmse = []
    shifts = [-2, -1, 0, 1, 2]
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
                try:
                    tmp_mdl = sm.tsa.statespace.SARIMAX(y_train,
                                                        order = param,
                                                        seasonal_order = param_seasonal,
                                                        enforce_stationarity=True,
                                                        enforce_invertibility=True)
                    res = tmp_mdl.fit()
                    forecast = res.forecast(len(X_test))
                    y_hat = forecast
                    y_true = y_test
                    best_rmse = 100000000000000000000000
                    for shift in shifts:
                        mse = ((y_hat.shift(shift) - y_true) ** 2).mean()
                        rmse = math.sqrt(mse)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_shift = shift
                    
                    model_data = {'model': '%sx%s' % (param, param_seasonal), 'aic': res.aic,
                                  'rmse': best_rmse, 'shift': best_shift}
                    if not best_seasonal_pdq_aic:
                        best_seasonal_pdq_aic.append(model_data)
                    else:
                        for i, model in enumerate(best_seasonal_pdq_aic):
                            if model['aic'] > model_data['aic']:
                                best_seasonal_pdq_aic.insert(i, model_data)
                                best_seasonal_pdq_aic.pop()
                                break

                            if len(best_seasonal_pdq_aic)  < 3:
                                best_seasonal_pdq_aic.append(model_data)
                    if not best_seasonal_pdq_rmse:
                        best_seasonal_pdq_rmse.append(model_data)
                    else:                            
                        for i, model in enumerate(best_seasonal_pdq_rmse):
                            if model['rmse'] > model_data['rmse']:
                                best_seasonal_pdq_rmse.insert(i, model_data)
                                best_seasonal_pdq_rmse.pop()
                                break

                            if len(best_seasonal_pdq_rmse) < 3:
                                best_seasonal_pdq_rmse.append(model_data)

                except Exception as e:
                    print("Unexpected error: ", e)
                    continue
    end = time.time()
    print('Duration: %s' % (end-start))
    return {'best_aics': best_seasonal_pdq_aic,'best_rmses': best_seasonal_pdq_rmse}

#a = test_params(maxrange=2, article='felco2')
#b = test_params(maxrange=4, article='felco2')

a = test_params(maxrange=2)
b = test_params(maxrange=4)


# In[ ]:



print('Maxrange 2 - Best AIC\'s')
for elem in a['best_aics']:
    print('Model: {} - AIC: {:.2f} - RMSE(Shift {}): {:.2f}'.format(elem['model'],elem['aic'],elem['shift'],elem['rmse']))
    
print('Maxrange 2 - Best RMSE\'s')
for elem in a['best_rmses']:
    print('Model: {} - AIC: {:.2f} - RMSE(Shift {}): {:.2f}'.format(elem['model'],elem['aic'],elem['shift'],elem['rmse']))

print('Maxrange 4 - Best AIC\'s')
for elem in b['best_aics']:
    print('Model: {} - AIC: {:.2f} - RMSE(Shift {}): {:.2f}'.format(elem['model'],elem['aic'],elem['shift'],elem['rmse']))
    
print('Maxrange 4 - Best RMSE\'s')
for elem in b['best_rmses']:
    print('Model: {} - AIC: {:.2f} - RMSE(Shift {}): {:.2f}'.format(elem['model'],elem['aic'],elem['shift'],elem['rmse']))


# In[44]:



def plot_forecast(param, param_seasonal, y_train, X_test, y_test):
    model = sm.tsa.statespace.SARIMAX(y_train,
                                      order = param,
                                      seasonal_order = param_seasonal,
                                      enforce_stationarity=True,
                                      enforce_invertibility=True)
    model_fitted = model.fit(disp=False)
    forecast = model_fitted.forecast(len(X_test))

    # compute the mean square error
    y_hat = forecast
    y_true = y_test
    mse = ((y_hat - y_true) ** 2).mean()
    print('Prediction quality: {:.2f} RMSE'.format(math.sqrt(mse)))
    print('AIC: {:.2f} RMSE'.format((model_fitted.aic)))
    
    lastyear = y[date(2016,1,1):date(2016,12,31)]
    plt.figure(figsize=(20,5))
    plt.plot(y_test, color='blue')
    plt.plot(forecast, color='red')
    plt.show()

    plt.figure(figsize=(20,5))
    plt.plot(y.index, y)

    X_train = y_train.index.map(dt.datetime.toordinal).values.reshape(-1, 1)
    y_train = y_train
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    prediction = regr.predict(X_train)
    plt.plot(X_train, prediction)


    years = [startdate, date(2015, 1, 1), date(2016, 1, 1), date(2017, 1, 1)]
    for year in years:
        plt.axvline(x=year, color='black')

    plt.plot(forecast, color='red')
    plt.show()


plot_forecast((0, 1, 0), (2, 3, 0, 12), y_train, X_test, y_test)

