
# coding: utf-8

# In[33]:


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


# In[34]:


default_startdate = date(2015, 10, 1)
default_enddate = date(2018, 10, 1)

# performs Augmented Dickey Fuller test
# params: training set data, current value for d (count of differencing)
# returns Tupel of stationary data and d
def dickey_fuller(data, d):
    dftest = adfuller(data, autolag='AIC')
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
        if (d >= 2):
            return (data, d)
    return (data, d)


# get ACF values for y, find seasonality pattern
# returns calculated Vlaue for S
def getSeasonality(series):
    result = seasonal_decompose(series, model='additive')
    #print(result.trend)
    #print(result.seasonal)
    #print(result.resid)
    #print(result.observed)
    
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
                print('Model (%s)x(%s) - AIC: %s' % (param, param_seasonal, res.aic))
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_mdl = tmp_mdl
            except:
                # print("Unexpected error:", sys.exc_info()[0])
                continue

    #print("Best SARIMAX{}x{} model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
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
    S = getSeasonality(y_train)
    D = getSeasonalDifferencing(y_train, S)
    
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
    #param = (p, d, q)
    #param_seasonal = (P, D, Q, S)
    #return sm.tsa.statespace.SARIMAX(y_train,
    #                                 order = param,
    #                                 seasonal_order = param_seasonal,
    #                                 enforce_stationarity=False,
    #                                 enforce_invertibility=False)

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


# In[35]:


ids = [722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830]
#ids = [722, 3986, 7979, 7612]
start = date(2015, 10, 1)
end = date(2018, 10, 1)
df = get_dataframe()


# In[38]:


rmses = np.array([])
articleIds = df['ArtikelID'].unique()

sum_article = pd.DataFrame(columns=['SUM', 'ArtikelID'])

for articleId in articleIds:
    dfTemporary = df[(df['ArtikelID'].isin([articleId]))]
    dfTemporary = dfTemporary.drop(columns=['ArtikelID'])
    dfTemporary = group_by_frequence(dfTemporary, frequence='MS',startdate=start, enddate=end)
    menge = dfTemporary['Menge']
    results = {'ArtikelID': articleId, 'SUM': menge.sum()}
    sum_article = sum_article.append(results, ignore_index=True)

sum_article = sum_article.set_index('ArtikelID')
sum_article = sum_article.sort_values('SUM', ascending=False)
top50_articles = sum_article.head(50)
display(top50_articles)

