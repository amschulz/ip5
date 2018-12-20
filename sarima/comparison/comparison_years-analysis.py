
# coding: utf-8

# In[5]:


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


# In[6]:


default_startdate = date(2015, 1, 1)
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
    return D

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
    
    for param in params:
        d = param[1]
        if d >= 3:
            continue
        for param_seasonal in seasonal_params:
            D = param_seasonal[1]
            if D:
                param = (param[0], 0, param[2])
                d = 0
            if (D + d) >= 3:
                continue
            try:
                tmp_mdl = sm.tsa.statespace.SARIMAX(y_train, 
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
        
     # evaluate differencing, Seasonality and Seasonal Diff.
    xx, d = dickey_fuller(y_train, 0)
    if xx is None and d is None:
        return {'model': {}}
    S = getSeasonality(y_train)
    D = getSeasonalDifferencing(y_train, S)
    if not D:
        return {'model': {}}
    
    # id d + D > 2 then set d = 0
    if (d+D >2): d=0
    known_params['d'] = d
    known_params['D'] = D
    known_params['S'] = S
    
    if S != 12:
        return {'model': {}}

    results = eval_pdq_by_example(maxparam=3, y_train=y_train, **known_params)
    results['Known Params'] = ' '.join(list(known_params.keys()))
    return results

def get_quality(test, forecast):
    percantege_error = []
    test_without_0 = test[test != 0.0]
    default_for_0 = np.min(test_without_0) if len(test_without_0) > 6 else 1
    for i,v in enumerate(test):
        # print('Test-Value %s - Forecast-Value %s' % (v, forecast[i]))
        percantege_error.append(math.sqrt(((forecast[i] / max(v, default_for_0)) - 1)**2))
    mape = sum(percantege_error) / float(len(percantege_error))
    median = statistics.median(percantege_error)
    return {'mape': mape,
            'median': median}


# In[7]:


start = date(2015, 1, 1)
end = date(2018, 10, 1)
df = get_dataframe()


# In[ ]:


import datetime
indices = [982,6058,3188,2818,3212,6364,4740,3597,4739,1651,239,4737,3337,136,1060,4175,4738,6365,4861,88,4863,3218,5148,
           8109,5671,2276,3139,8554,876,4741,3453,1519,5080,3229,3986,4791,4948,4980,1061,92,4981,91,4890,677,1653,1649,93,
           4250,3335,3582]
years_to_compare = range(3,11)
for ye in years_to_compare:
    ty = 2018
    s = date(ty - ye, 1, 1)
    ed = date(ty, 10, 1)
    
    df = get_dataframe(indices)
    rmses = np.array([])
    articleIds = df['ArtikelID'].unique()
    results_df = pd.DataFrame(columns=['ArtikelID', 'Skipped', 'Error', 'Mape', 'Median', 'TrainData Mean', 'aic', 'Known Params',
                                       'p','d','q','P','D','Q','S',
                                      ])
    for articleId in articleIds:
        dfTemporary = df[(df['ArtikelID'].isin([articleId]))]
        dfTemporary = dfTemporary.drop(columns=['ArtikelID'])
        dfTemporary = group_by_frequence(dfTemporary, frequence='MS',startdate=s, enddate=ed)
        dfTemporary = dfTemporary.drop(columns=['Datum'])
        y = dfTemporary['Menge']
        y_train = y[:date(2017,12,31)]
        X_train = y_train.index
        y_test = y[date(2018,1,1):date(2018,10,28)]
        X_test = y_test.index
        res = get_best_model(y_train)
        model = res['model']
        if (model is None) or ( not model):
            display('Skipped %s' % articleId)
            results_df = results_df.append({'ArtikelID': articleId, 
                                            'Skipped': True,
                                            'Error': False},
                                           ignore_index=True)
            continue
        try:
            fitted_model = model.fit()
            pred = fitted_model.forecast(dynamic=True, steps=10)
            for i,v in enumerate(pred):
                if v < 0:
                    pred[i] = 0
        except Exception as e:
            print('Exception at Article %s: %s' % (articleId, e))
            print(model)
            results_df = results_df.append({'ArtikelID': articleId, 
                                            'Skipped': False,
                                            'Error': True},
                                           ignore_index=True)
            continue

        quality = get_quality(y_test, pred)
        results = {'ArtikelID': articleId, 
                                        'Skipped': False,
                                        'Error': False, 
                                        'Median': quality['median'],
                                        'Mape': quality['mape'],
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
        rmses = np.append(rmses, [quality['mape'] / y_train.mean()])
    results_df = results_df.set_index('ArtikelID')
    print('Mape-Mean %s: %s' % (results_df['Mape'].mean(),ye))
    print('Median-Mean %s: %s' % (results_df['Median'].mean(),ye))
    display(results_df)
    outputfile = 'SARIMA_Pipeline_%s_Jahre_v2.csv' % (ye)
    results_df.to_csv(path_or_buf='outputs/%s' % outputfile,
                         sep=';')

