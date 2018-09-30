
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import date
from statsmodels.tsa.stattools import adfuller
import math
import itertools


# In[9]:


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
    param_Q = [D] if Q != -1 else range(0,maxparam)
    param_S = [S] if S != -1 else range(0,maxparam)
    
    
    params = list(itertools.product(param_p, param_d, param_q))
    seasonal_params = list(itertools.product(param_P, param_D, param_Q, param_S))
    best_aic = 1000000000000
    best_pdq = None
    best_seasonal_pdq = None
    best_mdl = None
    
    for param in params:
        for param_seasonal in seasonal_params:
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

    #print("Best SARIMAX{}x{} model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
    return best_mdl
    
    

def get_best_model(y_train):
    known_params = {}
    p = 0
    d = 0
    q = 0
    P = 2
    D = 0
    Q = 0
    S = 12
    y_train_stationary, d = getStationaryData(y_train)
    if y_train_stationary is None and d == -1:
        return None
    
    known_params['d'] = d
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
    #display(acf_peaks)
    #display(pacf_peaks)
    
    #plt.plot(acf_limit_below)
    #plt.plot(acf_limit_above)
    #plt.plot(acf_value, color='red')
    #plt.title('ACF')
    #plt.show()
    
    #plt.plot(pacf_limit_below)
    #plt.plot(pacf_limit_above)
    #plt.plot(pacf_value, color='red')
    #plt.title('PACF')
    #plt.show()


    if 12 in acf_peaks and 12 in pacf_peaks:
        known_params['S'] = 12
        known_params['p'] = 0
        known_params['d'] = 0
        known_params['q'] = 0
        
    else:
        known_params['S'] = 0
        known_params['P'] = 0
        known_params['D'] = 0
        known_params['Q'] = 0
    
    np_acf = np.array(acf_value)
    np_acf_square = np.square(np_acf)
    np_acf = np.sqrt(np_acf_square)
    differenced_acf = np.diff(np_acf[:-1])
    np_pacf = np.array(pacf_value)
    np_pacf_square = np.square(np_pacf)
    np_pacf = np.sqrt(np_pacf_square)
    differenced_pacf = np.diff(np_pacf[:-1])
    big_decrease_acf = np.where(differenced_acf < -0.15)
    big_decrease_pacf = np.where(differenced_pacf < -0.15)
    cutoff_x_acf = -1
    if len(big_decrease_acf) == 1 and big_decrease_acf[0][0] != 0:
        cutoff_x_acf = big_decrease_acf[0][0]
    elif len(big_decrease_acf) == 2:
        cutoff_x_acf = big_decrease_acf[0][1]
    if cutoff_x_acf != -1:
        for lag in range(0,cutoff_x_acf+1):
            if lag in acf_peaks:
                break
        else:
            if len(big_decrease_pacf) == 0:
                known_params['p'] = cutoff_x_acf
                known_params['q'] = 0

    cutoff_x_pacf = -1     
    if len(big_decrease_pacf) == 1 and big_decrease_pacf[0][0] != 0:
        cutoff_x_pacf = big_decrease_pacf[0][0]
    elif len(big_decrease_pacf) == 2:
        cutoff_x_pacf = big_decrease_pacf[0][1]
    if cutoff_x_pacf != -1:
        for lag in range(0,cutoff_x_pacf+1):
            if lag in pacf_peaks:
                break
        else:
            if len(big_decrease_acf) == 0:
                known_params['p'] = 0
                known_params['q'] = cutoff_x_pacf
        

        
    # display(known_params)
    return eval_pdq_by_example(maxparam=4, y_train=y_train_stationary, **known_params)
    param = (p, d, q)
    param_seasonal = (P, D, Q, S)
    return sm.tsa.statespace.SARIMAX(y_train,
                                     order = param,
                                     seasonal_order = param_seasonal,
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)

def get_rmse(test, forecast):
    if len(test) != len(forecast):
        test = test[:-1]
    mse = ((test - forecast) ** 2).mean()
    return math.sqrt(mse)


# In[12]:


ids = [722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830]
start = date(2010, 1, 1)
end = date(2018, 1, 1)
df = get_dataframe()


# In[ ]:


rmses = np.array([])
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
    if model is None:
        display('Skipped %s' % articleId)
        continue
    try:
        fitted_model = model.fit()
        #pred = fitted_model.get_prediction(dynamic=False,
        #                                    steps=12,
        #                                   start=pd.to_datetime('2016-12-01'),
        #                                   end=pd.to_datetime('2017-12-31'))
        pred = fitted_model.forecast(dynamic=True, steps=12)
    except Exception as e:
        print('Exception at Article %s' % articleId)
        continue
    #plt.figure(figsize=(20,5))
    #plt.plot(pred, color='red')
    #plt.plot(y_test, color='blue')
    #plt.show()
    #rmses = np.append(rmses, [get_rmse(y_test, pred.predicted_mean) / y_train.mean()])
    #print('RMSE; %s' % get_rmse(y_test, pred))
    #print('Mean; %s' % y_train.mean())
    rmses = np.append(rmses, [get_rmse(y_test, pred) / y_train.mean()])
    # display('RMSE: %s - Mean: %s' % (get_rmse(y_test, pred), y_train.mean()))
display(rmses)
display('RMSE Mean: %s' % (np.mean(rmses)))


# In[62]:


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
    #pred = fitted_model.get_prediction(dynamic=False,
    #                                   steps=12,
    #                                   start=pd.to_datetime('2016-12-01'),
    #                                   end=pd.to_datetime('2017-12-31'))
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


# In[2]:


plt.plot([65,46,51])
plt.plot([50,96,68])
plt.show()


# In[158]:


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
    display(y_train)
    diffed = np.diff(y_train)
    display(np.asarray(diffed).shape[0])
    break

