
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
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
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
def dickey_fuller(y, d):
    dftest = adfuller(y, autolag='AIC')
    #dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    #for key, value in dftest[4].items():
    #    dfoutput['Critical Value ({})'.format(key)] = value
    #print(dfoutput)
    adf_val = dftest[0]

    if adf_val < dftest[4]['5%']:
        # print('Is stationary')
        return d
    else:
        #print('Is not stationary. Try differencing.')
        d = d+1
        d = dickey_fuller (np.diff(y), d)
        if (d >= 3):
            return d;
    return d


# In[3]:


def show_data(y):
    title='Data, Mean, Std'

    # monthly moving averages (12 month window)
    w = 12
    rolling_mean = pd.rolling_mean(y, window=w)
    rolling_std = pd.rolling_std(y, window=w)

    # setup layout
    fig = plt.figure(figsize=(14, 5))
    layout = (1, 2)  # rows, cols
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

    # time series plot
    y.plot(ax=ts_ax)
    rolling_mean.plot(ax=ts_ax, color='crimson');
    rolling_std.plot(ax=ts_ax, color='darkslateblue');
    plt.legend(loc='best')
    ts_ax.set_title(title, fontsize=24);

    plt.tight_layout();
    #plt.savefig('./img/{}.png'.format(filename))
    plt.show()
    
def show_acf_pacf(y, lags):
    # 1x diff
    y_diff1= np.diff(y)

    # setup layout
    fig = plt.figure(figsize=(14, 5))
    layout = (2, 2)  # rows, cols
    acf_ax = plt.subplot2grid(layout, (0, 0))
    pacf_ax = plt.subplot2grid(layout, (0, 1))
    acfd_ax = plt.subplot2grid(layout, (1, 0))
    pacfd_ax = plt.subplot2grid(layout, (1, 1))

    # acf and pacf
    acfd = sm.tsa.stattools.acf(y, nlags=lags)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5) 

    acf_ax.set_title('ACF Plot')
    pacf_ax.set_title('PACF Plot');

    # differenced acf/pacf
    acfd_2 = sm.tsa.stattools.acf(y_diff1, nlags=lags)
    smt.graphics.plot_acf(y_diff1, lags=lags, ax=acfd_ax, alpha=0.5)
    smt.graphics.plot_pacf(y_diff1, lags=lags, ax=pacfd_ax, alpha=0.5) 
    acfd_ax.set_title('Differenced ACF')
    pacfd_ax.set_title('Differenced PACF')

    plt.tight_layout();
    #plt.savefig('./img/{}.png'.format(filename))
    plt.show()

def show_charts(y):
    # setup layout
    fig = plt.figure(figsize=(14, 5))
    layout = (1, 2)  # rows, cols
    qq_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    
    # qq plot
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')

    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25);
    hist_ax.set_title('Histogram');
    
    plt.tight_layout();
    #plt.savefig('./img/{}.png'.format(filename))
    plt.show()


# In[4]:


enddate = date(2018, 1, 1)
startdate = date(2010, 1, 1)

# load data from file
#df = get_dataframe_SingleArt('gefaschlauch60','M')
#df = get_dataframe_SingleArt('felco2','M') # tailing off
df = get_dataframe_SingleArt('bambus1201012','M')
# Best SARIMAX(0, 1, 1)x(0, 2, 1, 12) model - AIC:1317.2675177190258


# prepare data
df = df.drop(columns=['Datum'])
y = df['Menge']

# make train and test set
splitdate = date(2016, 12, 31)
y_train = y[:splitdate]
y_test = y[splitdate:]
y_test = y_test[1:]
X_test = y_test.index.map(dt.datetime.toordinal).values.reshape(-1, 1)


# In[5]:


# show_data(y_train)
show_acf_pacf(y_train, 24)
#show_charts(y_train)


# In[37]:


# https://people.duke.edu/~rnau/arimrule.htm
# applying the rules:

lags = 24
acf  = sm.tsa.stattools.acf(y_train, nlags=lags)
pacf = sm.tsa.stattools.acf(y_train, nlags=lags)

#Identifying the order of differencing and the constant:
#
#Rule 1: 
#  If the series has positive autocorrelations out to a high number of lags (say, 10 or more), 
#  then it probably needs a higher order of differencing.

df = dickey_fuller(y_train, 0)

#Rule 2: 
#  If the lag-1 autocorrelation is zero or negative, or the autocorrelations are all small and 
#  patternless, then the series does not need a higher order of differencing. If the 
#  lag-1 autocorrelation is -0.5 or more negative, the series may be overdifferenced.  
#  !!! BEWARE OF OVERDIFFERENCING.

da = df
if (acf[0]<=0):
    da = da+1
else:
    counter=0
    td = 0
    for value in acf:
        if (value>0.25):
            counter = counter + 1
    if (counter == 0):
        da=da+1
    

#Rule 3: 
#  The optimal order of differencing is often the order of differencing at which the 
#  standard deviation is lowest. (Not always, though. Slightly too much or slightly too 
#  little differencing can also be corrected with AR or MA terms. See rules 6 and 7.)

y_diff1 = np.diff(y_train)
y_diff2 = np.diff(y_diff1)

std0 = y_train.std()
std1 = y_diff1.std()
std2 = y_diff2.std()

if (std0<std1):
    if(da>0):
        d = (0, da)
    else:
        d = (da)
else:
    if(std1<std2):
        if(da == 0):
            d = (da,1)
        elif(da == 1):
                d = (da)
        else:
            d = (1,da)

print('### Differencing: ###')
print('Dickey Fuller: df = %i' % df)
print('DF and lag-1:  da = %i' % da)
print('std of original series      ', std0)
print('std of differenced data     ', std1)
print('std of differenced data     ', std2)
print('Result:        d  =', d)

#Rule 4: 
#  A model with no orders of differencing assumes that the original series is stationary 
#  (among other things, mean-reverting). A model with one order of differencing assumes 
#  that the original series has a constant average trend (e.g. a random walk or SES-type model, 
#  with or without growth). A model with two orders of total differencing assumes that the 
#  original series has a time-varying trend (e.g. a random trend or LES-type model).


#Rule 5: 
#  A model with no orders of differencing normally includes a constant term (which allows for a 
#  non-zero mean value). A model with two orders of total differencing normally does not include 
#  a constant term. In a model with one order of total differencing, a constant term should be 
#  included if the series has a non-zero average trend.


# In[38]:


show_acf_pacf(y_train, 24)


# In[39]:


# Identifying the numbers of AR and MA terms:

acf_d  = sm.tsa.stattools.acf(y_diff1, nlags=lags)
pacf_d = sm.tsa.stattools.acf(y_diff1, nlags=lags)

# Rule 6: 
#   If the partial autocorrelation function (PACF) of the differenced series displays a 
#   sharp cutoff and/or the lag-1 autocorrelation is positive--i.e., if the series appears 
#   slightly "underdifferenced"--then consider adding one or more AR terms to the model. 
#   The lag beyond which the PACF cuts off is the indicated number of AR terms.

ar = 0
# lag 1 is positive -> underdifferenced -> add AR
if(pacf_d[1] > 0):
    ar = 1
# sharp cutoff after lag c -> AR = c


# Rule 7: 
#   If the autocorrelation function (ACF) of the differenced series displays a sharp 
#   cutoff and/or the lag-1 autocorrelation is negative--i.e., if the series appears 
#   slightly "overdifferenced"--then consider adding an MA term to the model. The lag 
#   beyond which the ACF cuts off is the indicated number of MA terms.

ma = 0
# lag 1 is negative -> overdifferenced -> add MA
if(acf_d[1] < 0):
    ma = 1
#sharp cutoff after lag c -> MA = c
    
    
# Rule 8: 
#   It is possible for an AR term and an MA term to cancel each other's effects, 
#   so if a mixed AR-MA model seems to fit the data, also try a model with one fewer 
#   AR term and one fewer MA term--particularly if the parameter estimates in the 
#   original model require more than 10 iterations to converge. 
#   !!!! BEWARE OF USING MULTIPLE AR TERMS AND MULTIPLE MA TERMS IN THE SAME MODEL.

# Rule 9: 
#   If there is a unit root in the AR part of the model--i.e., if the sum of the 
#   AR coefficients is almost exactly 1--you should reduce the number of AR terms 
#   by one and increase the order of differencing by one.

# Rule 10: 
#   If there is a unit root in the MA part of the model--i.e., if the sum of the 
#   MA coefficients is almost exactly 1--you should reduce the number of MA terms 
#   by one and reduce the order of differencing by one.

# Rule 11: 
#   If the long-term forecasts* appear erratic or unstable, there may be a unit root 
#   in the AR or MA coefficients.

print ('p = %i' % ar)
print ('q = %i' % ma)


# In[151]:


# Identifying the seasonal part of the model:

# Rule 12: 
#   If the series has a strong and consistent seasonal pattern, then you must use an order 
#   of seasonal differencing (otherwise the model assumes that the seasonal pattern will 
#   fade away over time). However, never use more than one order of seasonal differencing or 
#   more than 2 orders of total differencing (seasonal+nonseasonal).

# Rule 13: 
#   If the autocorrelation of the appropriately differenced series is positive at lag s, 
#   where s is the number of periods in a season, then consider adding an SAR term to the 
#   model. If the autocorrelation of the differenced series is negative at lag s, consider 
#   adding an SMA term to the model. The latter situation is likely to occur if a seasonal 
#   difference has been used, which should be done if the data has a stable and logical 
#   seasonal pattern. The former is likely to occur if a seasonal difference has not been 
#   used, which would only be appropriate if the seasonal pattern is not stable over time. 
#   You should try to avoid using more than one or two seasonal parameters (SAR+SMA) in 
#   the same model, as this is likely to lead to overfitting of the data and/or problems 
#   in estimation.


# In[27]:


def eval_pdq_by_example(y, p, d, q, S):
    # p: list of 0, 1, 2 OR [0]
    # d: list of 0, 1, 2 OR [0]
    # q: list of 0, 1, 2 OR [0]
    
    # one of q OR q must be [0]
    if (len(p)>1):
        q=[0]
    
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
            # skip, if d and D sum up to more than 2.
            if(param[1] + param_seasonal[1] < 3):
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
    
def evalSarima(y, p, d, q, P, D, Q, S):
    param = (p, d, q)
    param_seasonal = (P, D, Q, S)

    model = sm.tsa.statespace.SARIMAX(y,
                                      order = param,
                                      seasonal_order = param_seasonal,
                                      enforce_stationarity=True,
                                      enforce_invertibility=True)
    res = model.fit(disp=False)
    print(res.summary())

    res.plot_diagnostics(figsize=(16, 10))
    plt.tight_layout()
    plt.show()
    return res
    
def predictModel (mFit, data):
    # in-sample-prediction and confidence bounds
    # prediction time range
    pdate_start = pd.to_datetime('2016-12-31')
    pdate_end   = pd.to_datetime('2017-12-31')
    chart_start = pd.to_datetime('2010-12-31')

    pred = mFit.get_prediction(start=pdate_start, 
                              end=pdate_end,
                              dynamic=True)
    pred_ci = pred.conf_int()

    plt.figure(figsize=(20,10))
    # plot in-sample-prediction
    yy = data['Menge']
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


# In[31]:


eval_pdq_by_example(y_train, [0,1,2], [0,1], [0,1,2], 6)
# Best SARIMAX(1, 0, 0)x(2, 1, 0, 6) model - AIC:444.53569967170387


# In[40]:


eval_pdq_by_example(y_train, [0,1,2], [0,1], [0,1,2], 12)
# d, D = 0, 1
# p, q = 0, 1
#Best SARIMAX(0, 1, 1)x(0, 2, 1, 12) model - AIC:1317.2675177190258

