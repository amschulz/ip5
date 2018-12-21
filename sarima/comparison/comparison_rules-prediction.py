
# coding: utf-8

# In[140]:


# coding: utf-8
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
import datetime
from operator import itemgetter


# In[144]:


# Check if acf/pacf has a cutoff at a specific lag
def get_data_behavior(cf):
    diffed_cf = np.diff(cf)
    big_decrease_lags = np.where(diffed_cf < -0.15)
    cutoff_at = None
    if big_decrease_lags[0][0] == 0:
        if len(big_decrease_lags) == 2:
            return ('CUTOFF', big_decrease_lags[0][1])
    else:
        if len(big_decrease_lags) == 1:
            return ('CUTOFF', big_decrease_lags[0][0])

    
    return ('NOPATTERN', 0)

def rule_01(known_ps, da):
    #Rule 1: 
    #  If the series has positive autocorrelations out to a high number of lags (say, 10 or more), 
    #  then it probably needs a higher order of differencing.
    acf = sm.tsa.stattools.acf(da, nlags=36)
    first_negative_at_index = -1
    for i, val in enumerate(acf):
        if val <= 0:
            first_negative_at_index = i
            break
    if first_negative_at_index > 10:
        try:
            known_ps['d'] = known_ps.get('d', 0) + 1
            da = np.diff(da)
        except Error as e:
            print('Error applying rule_01')
    return (known_ps, da)

def rule_02(known_ps, da):
    #Rule 2: 
    #  If the lag-1 autocorrelation is zero or negative, or the autocorrelations are all small and 
    #  patternless, then the series does not need a higher order of differencing. If the 
    #  lag-1 autocorrelation is -0.5 or more negative, the series may be overdifferenced.  
    #  !!! BEWARE OF OVERDIFFERENCING.
    acf = sm.tsa.stattools.acf(da, nlags=36)
    if acf[1] <= -0.5:
        known_ps['d'] = known_ps.get('d',0) - 1
    return (known_ps, da)

def rule_03(known_ps, da):
    #Rule 3: 
    #  The optimal order of differencing is often the order of differencing at which the 
    #  standard deviation is lowest. (Not always, though. Slightly too much or slightly too 
    #  little differencing can also be corrected with AR or MA terms. See rules 6 and 7.)
    
	# see apply_rules
    return (known_ps, da)

def rule_04(known_ps, da):
    #Rule 4: 
    #  A model with no orders of differencing assumes that the original series is stationary 
    #  (among other things, mean-reverting). A model with one order of differencing assumes 
    #  that the original series has a constant average trend (e.g. a random walk or SES-type model, 
    #  with or without growth). A model with two orders of total differencing assumes that the 
    #  original series has a time-varying trend (e.g. a random trend or LES-type model).
    
    # Nothing to do
    return (known_ps, da)


def rule_05(known_ps, da):
    #Rule 5: 
    #  A model with no orders of differencing normally includes a constant term (which allows for a 
    #  non-zero mean value). A model with two orders of total differencing normally does not include 
    #  a constant term. In a model with one order of total differencing, a constant term should be 
    #  included if the series has a non-zero average trend.
    
    # Nothing to do
    return (known_ps, da)

def rule_06(known_ps, da):
    # Rule 6: 
    #   If the partial autocorrelation function (PACF) of the differenced series displays a 
    #   sharp cutoff and/or the lag-1 autocorrelation is positive--i.e., if the series appears 
    #   slightly "underdifferenced"--then consider adding one or more AR terms to the model. 
    #   The lag beyond which the PACF cuts off is the indicated number of AR terms.
    pacf = sm.tsa.stattools.pacf(da, nlags=36)
    typ, lag = get_data_behavior(pacf)
    if pacf[1] > 0 and typ == 'CUTOFF':
        ar = lag + 1
        known_ps['p'] = ar
        plt.show()
    return (known_ps, da)


def rule_07(known_ps, da):
    # Rule 7: 
    #   If the autocorrelation function (ACF) of the differenced series displays a sharp 
    #   cutoff and/or the lag-1 autocorrelation is negative--i.e., if the series appears 
    #   slightly "overdifferenced"--then consider adding an MA term to the model. The lag 
    #   beyond which the ACF cuts off is the indicated number of MA terms.
    acf = sm.tsa.stattools.acf(da, nlags=36)
    typ, lag = get_data_behavior(acf)
    if acf[1] < 0 and typ == 'CUTOFF':
        ma = lag + 1
        known_ps['q'] = ma
        plt.show()
    return (known_ps, da)

def rule_08(known_ps, da):
    # Rule 8: 
    #   It is possible for an AR term and an MA term to cancel each other's effects, 
    #   so if a mixed AR-MA model seems to fit the data, also try a model with one fewer 
    #   AR term and one fewer MA term--particularly if the parameter estimates in the 
    #   original model require more than 10 iterations to converge. 
    #   !!!! BEWARE OF USING MULTIPLE AR TERMS AND MULTIPLE MA TERMS IN THE SAME MODEL.
    
    # Already in iteration through params in eval_pdq_by_example.
    pass

def rule_09(known_ps, da):
    # Rule 9: 
    #   If there is a unit root in the AR part of the model--i.e., if the sum of the 
    #   AR coefficients is almost exactly 1--you should reduce the number of AR terms 
    #   by one and increase the order of differencing by one.
    
    # No time to implement. Had to be done in eval_pdq_by_example
    pass

def rule_10(known_ps, da):
    # Rule 10: 
    #   If there is a unit root in the MA part of the model--i.e., if the sum of the 
    #   MA coefficients is almost exactly 1--you should reduce the number of MA terms 
    #   by one and reduce the order of differencing by one.
    
    # No time to implement. Had to be done in eval_pdq_by_example
    pass

def rule_11(known_ps, da):
    # Rule 11: 
    #   If the long-term forecasts* appear erratic or unstable, there may be a unit root 
    #   in the AR or MA coefficients.
    
    # Nothing to do.
    pass

def rule_12(known_ps, da):
    # Rule 12: 
    #   If the series has a strong and consistent seasonal pattern, then you must use an order 
    #   of seasonal differencing (otherwise the model assumes that the seasonal pattern will 
    #   fade away over time). However, never use more than one order of seasonal differencing or 
    #   more than 2 orders of total differencing (seasonal+nonseasonal).
    
    # Done before calling apply_rules.
    pass

def rule_13(known_ps, da):
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
    acf = sm.tsa.stattools.acf(da, nlags=36)
    if known_ps['D'] > 0:  
        if acf[known_ps['S']] > 0:
            known_ps['P'] = known_ps.get('P',0) + 1
        if acf[known_ps['S']] < 0:
            known_ps['Q'] = known_ps.get('Q',0) + 1        
    return (known_ps, da)
    
# apply the rules 1 to 13. See above for more details.
def apply_rules(known_param, traindata):
	# appyling of rule 3
    diffed0 = traindata
    diffed1 = np.diff(diffed0)
    l = [(0,diffed0, 0, diffed0.std()), (1,diffed1, 0, diffed1.std())]    

    if known_param['D'] == 0:
        diffed2 = np.diff(diffed1)
        l.append((2, diffed2, 0, diffed2.std()))
    l.sort(key=lambda das: das[3])
	
	
    for ds in l:
        errors = ds[2]
        known_param['d'] = ds[0]
        r1, d1 = rule_01(known_param, ds[1])
        r2, d2 = rule_02(known_param, ds[1])
        if r1['d'] != known_param['d']: errors += 2
        if r2['d'] != known_param['d']: errors += 1
        if r1['d'] + known_param['D'] > 2: errors += 4
        if r2['d'] < 0: errors += 8
        ds = (ds[0], ds[1], errors)
    known_param['d'] = l[0][0]

    traindata = l[0][1]
    oldd = known_param['d']
    known_param, traindata = rule_06(known_param, traindata)        
    known_param, traindata = rule_07(known_param, traindata)
    known_param, traindata = rule_13(known_param, traindata)
    return known_param


# In[145]:


default_startdate = date(2010, 1, 1)
default_enddate = date(2018, 10, 1)

# performs Augmented Dickey Fuller test
# params: training set data, current value for d (count of differencing)
# returns Tupel of stationary data and d
def dickey_fuller(data, d):
    try:
        dftest = adfuller(data, autolag='AIC')
    except ValueError as e:
        return (None, None)
    adf_val = dftest[0]
    
    if math.isnan(adf_val):
        return (data, 0)
    
    if adf_val < dftest[4]['5%']:
        # is stationary
        return (data, d)
    else:
        #not stationary -> differencing.
        data = np.diff(data)
        d = d+1
        data, d = dickey_fuller (data, d)
        # limit to 2, there should never be more than 2x differencing
        # results in loss of quality for prediction
        if (d and d >= 2):
            return (data, d)
    return (data, d)


# get ACF values for y, find seasonality pattern
# returns calculated Vlaue for S
def getSeasonality(series):
    result = seasonal_decompose(series, model='additive')
    
    seasonal = [round(j*100) for i, j in enumerate(result.seasonal)]
    peak = max(seasonal)
    res = [i for i, j in enumerate(seasonal) if j == peak]

    S = 0
    if len(res)>1 :
        S=res[1]-res[0]
        
    return S
    
# calculate differenced Seasonal series and perform Dickey Fuller
# Returns seasonal differencing D
def getSeasonalDifferencing (series, S):
    seasonal_series = np.array([])
    max_l = len(series)
    i = 0
    while max_l > (i+S):
        seasonal_series = np.append(seasonal_series, [series[i+S]-series[i]])
        i = i + 1
    
    data, D = dickey_fuller(seasonal_series, 0)
    return data, D

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

def getStationaryData(y_train):
    drange = range(0,3)
    y_train_stationary = y_train
    for d in drange:
        dftest = adfuller(y_train_stationary, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value ({})'.format(key)] = value
        adf_val = dftest[0]

        if adf_val < dftest[4]['5%']:
            return (y_train_stationary, d)
        else:
            y_train_stationary = np.diff(y_train_stationary)
    return (None, -1)

# GridSearch Algorithm. Known parameters can be given as parameter and will not be changed. 
def eval_pdq_by_example(maxparam,
                        y_train,
                        p=-1,d=-1,q=-1,
                        P=-1,D=-1,Q=-1,S=-1):
    param_p = [p] if p != -1 else range(0,maxparam)
    param_d = [d] if d != -1 else range(0,maxparam)
    param_q = [q] if q != -1 else range(0,maxparam)
    param_P = [P] if P != -1 else range(0,maxparam)
    param_D = [D] if D != -1 else range(0,maxparam)
    param_Q = [Q] if Q != -1 else range(0,maxparam)
    param_S = [S] if S != -1 else range(0,maxparam)
    
    
    params = list(itertools.product(param_p, param_d, param_q))
    seasonal_params = list(itertools.product(param_P, param_D, param_Q, param_S))
    best_aic = 1000000000000
    best_pdq = (-1,-1,-1)
    best_seasonal_pdq = (-1,-1,-1,-1)
    best_mdl = None
    
	# iterate through all non-seasonal and seasonal paramter-combinations.
    for param in params:
        for param_seasonal in seasonal_params:
            if param_seasonal[3] != 0:
                if param_seasonal[0] + param_seasonal[2] > 2:
                    continue
            try:
                tmp_mdl = sm.tsa.statespace.SARIMAX(y_train, 
                                                    order = param,
                                                    seasonal_order = param_seasonal,
                                                    enforce_stationarity=True,
                                                    enforce_invertibility=True)
                res = tmp_mdl.fit()
				
				# check if model is better than current best model and replace if so.
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_mdl = tmp_mdl
            except:
                continue

    return {'model': best_mdl,
            'aic': best_aic,
            'paramter': {'p': best_pdq[0],
                         'd': best_pdq[1],
                         'q': best_pdq[2],
                         'P': best_seasonal_pdq[0],
                         'D': best_seasonal_pdq[1],
                         'Q': best_seasonal_pdq[2],
                         'S': best_seasonal_pdq[3]}
           }
    
    

def get_best_model(y_train):
    known_params = {}
    p = 0
    d = 0
    q = 0
    P = 1
    D = 0
    Q = 0
    S = 12
    
	# get seasonality and the seasonal differencing
    S = getSeasonality(y_train)
    xx2, d2 = getSeasonalDifferencing(y_train, S)
    known_params['D'] = d2
    known_params['S'] = S
    if S != 12:
        return {'model': {}}
    
    known_params = apply_rules(known_params, xx2)

    results = eval_pdq_by_example(maxparam=3, y_train=y_train, **known_params)
    results['Known Params'] = ' '.join(list(known_params.keys()))
    return results


# In[148]:


start = date(2010, 1, 1)
end = date(2018, 10, 1)
ids = [722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830]
df = get_dataframe()

rmses = np.array([])
articleIds = df['ArtikelID'].unique()
results_df = pd.DataFrame(columns=['ArtikelID', 'Skipped', 'Error', 'TrainData Mean', 'aic', 'Known Params',
                                   'p','d','q','P','D','Q','S',
                                  ])
for articleId in articleIds:
    dfTemporary = df[(df['ArtikelID'].isin([articleId]))]
    dfTemporary = dfTemporary.drop(columns=['ArtikelID'])
    dfTemporary = group_by_frequence(dfTemporary, frequence='MS',startdate=start, enddate=end)
    dfTemporary = dfTemporary.drop(columns=['Datum'])
	
	# Split in train and test data.
    y = dfTemporary['Menge']
    y_train = y[:date(2017,12,31)]
    X_train = y_train.index
    y_test = y[date(2018,1,1):date(2018,10,28)]
    X_test = y_test.index
	
	# get best possible model
    res = get_best_model(y_train)
    model = res['model']
	
	# Skip if model is not set
    if model is None:
        display('Skipped %s' % articleId)
        results_df = results_df.append({'ArtikelID': articleId, 
                                        'Skipped': True,
                                        'Error': False},
                                       ignore_index=True)
        continue
    try:
		# fit model and get prediction
        fitted_model = model.fit()

        pred = fitted_model.forecast(dynamic=True, steps=10)
        for i,v in enumerate(pred):
            if v < 0:
                pred[i] = 0
    except Exception as e:
        print('Exception at Article %s' % articleId)
        results_df = results_df.append({'ArtikelID': articleId, 
                                        'Skipped': False,
                                        'Error': True},
                                       ignore_index=True)
        continue

    results = {'ArtikelID': articleId, 
                                    'Skipped': False,
                                    'Error': False, 
                                    'TrainData Mean': y_train.mean(),
                                    'aic': res['aic'],
                                    'Known Params': res['Known Params']
                                    }
    
    # add data to results df
    for i, x in enumerate(y_test):
        results['Test M%s' % (i + 1)] = x
    for i, x in enumerate(pred):
        results['Forecast M%s' % (i + 1)] = x 
    results.update(res['paramter'])
    results_df = results_df.append(results,
                                   ignore_index=True)
results_df = results_df.set_index('ArtikelID')


# In[149]:

# display results and save them in a new file
display(results_df)
outputfile = 'SARIMA_Pipeline_%s.csv' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
results_df.to_csv(path_or_buf='outputs/%s' % outputfile,
                     sep=';')

