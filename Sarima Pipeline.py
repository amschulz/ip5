
# coding: utf-8

# In[142]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import date
from statsmodels.tsa.stattools import adfuller
import math
import itertools


# In[156]:


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
    best_pdq = (-1,-1,-1)
    best_seasonal_pdq = (-1,-1,-1,-1)
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
    y_train_stationary, d = getStationaryData(y_train)
    if y_train_stationary is None and d == -1:
        return {'model': None}
    
    known_params['d'] = d
    # get acf-data and pacf-data. The critical borders are also part of the return value!
    acf = sm.tsa.stattools.acf(y_train_stationary, nlags=36, alpha=0.1)
    pacf = sm.tsa.stattools.pacf(y_train_stationary, nlags=36, alpha=0.1)
    
    # for every element in acf[1]: Subtract the actual-value of the element in acf[1]
    # To get the critical values above and below for the acf-values
    acf_value = acf[0]
    acf_limit_below = [(x[0] - acf_value[i]) for i, x in enumerate(acf[1].tolist())]
    acf_limit_above = [(x[1] - acf_value[i]) for i, x in enumerate(acf[1].tolist())]
    # Same for pacf
    pacf_value = pacf[0]
    pacf_limit_below =[(x[0] - pacf_value[i]) for i, x in enumerate(pacf[1].tolist())]
    pacf_limit_above = [(x[1] - pacf_value[i]) for i, x in enumerate(pacf[1].tolist())]
    
    # get all lags where the acf value is above (+) or below (-) of the critical value.
    acf_peaks = [i for i, x in enumerate(acf_value) if (x < acf_limit_below[i] or x > acf_limit_above[i])]
    pacf_peaks = [i for i, x in enumerate(pacf_value) if (x < pacf_limit_below[i] or x > pacf_limit_above[i])]
    # Don't use peak 0 because it's always 1
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
        known_params['q'] = 0
        
    else:
        known_params['S'] = 0
        known_params['P'] = 0
        known_params['D'] = 0
        known_params['Q'] = 0
    
    # Create numpy array out of the acf-values
    np_acf = np.array(acf_value)
    # square it to remove negative values and take the square-root afterwards
    np_acf_square = np.square(np_acf)
    np_acf = np.sqrt(np_acf_square)
    
    # difference it so we can search for big decreases ("cut-off")
    differenced_acf = np.diff(np_acf[:-1])
    
    # Same process for pacf
    np_pacf = np.array(pacf_value)
    np_pacf_square = np.square(np_pacf)
    np_pacf = np.sqrt(np_pacf_square)
    differenced_pacf = np.diff(np_pacf[:-1])
    
    # look for big decreases in acf-values
    big_decrease_acf = np.where(differenced_acf < -0.15)
    big_decrease_pacf = np.where(differenced_pacf < -0.15)
    cutoff_x_acf = -1
    
    # sometimes 0 is also in the big-decrease acf. We have to ignore it
    if len(big_decrease_acf) == 1 and big_decrease_acf[0][0] != 0:
        cutoff_x_acf = big_decrease_acf[0][0]
    elif len(big_decrease_acf) == 2:
        cutoff_x_acf = big_decrease_acf[0][1]
        
    # if there is a big decrease in acf
    if cutoff_x_acf != -1:
        # check if all lags before the cutoff-value are above the critical values.
        for lag in range(0,cutoff_x_acf+1):
            if lag not in acf_peaks:
                break
        else:
            if len(big_decrease_pacf) == 0:
                print('---------------SET PQ 1')
                known_params['q'] = cutoff_x_acf
                known_params['p'] = 0

    # same for pacf values
    cutoff_x_pacf = -1     
    if len(big_decrease_pacf) == 1 and big_decrease_pacf[0][0] != 0:
        cutoff_x_pacf = big_decrease_pacf[0][0]
    elif len(big_decrease_pacf) == 2:
        cutoff_x_pacf = big_decrease_pacf[0][1]
    if cutoff_x_pacf != -1:
        for lag in range(0,cutoff_x_pacf+1):
            if lag not in pacf_peaks:
                break
        else:
            if len(big_decrease_acf) == 0:
                print('---------------SET PQ 2')
                known_params['q'] = 0
                known_params['p'] = cutoff_x_pacf
    
    
    # display(known_params)
    results = eval_pdq_by_example(maxparam=4, y_train=y_train_stationary, **known_params)
    results['Known Params'] = ' '.join(list(known_params.keys()))
    return results
    #param = (p, d, q)
    #param_seasonal = (P, D, Q, S)
    #return sm.tsa.statespace.SARIMAX(y_train,
    #                                 order = param,
    #                                 seasonal_order = param_seasonal,
    #                                 enforce_stationarity=False,
    #                                 enforce_invertibility=False)

def get_rmse(test, forecast):
    alternative_rmse_data = []
    test_without_0 = test[test != 0.0]
    default_for_0 = np.min(test_without_0) if len(test_without_0) > 6 else 1
    for i,v in enumerate(test):
        # print('Test-Value %s - Forecast-Value %s' % (v, forecast[i]))
        alternative_rmse_data.append(math.sqrt(((forecast[i] / max(v, default_for_0)) - 1)**2))

    alternative_rmse = sum(alternative_rmse_data) / float(len(alternative_rmse_data))
    # mse = ((test - forecast) ** 2).mean()
    # original_rmse = math.sqrt(mse)
    # print('alternative_rmse: %s - alternativ_rmse2: %s original_rmse: %s' %  (alternative_rmse, alternative_rmse2, original_rmse))
    return alternative_rmse


# In[160]:


ids = [722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830]
start = date(2010, 1, 1)
end = date(2018, 1, 1)
df = get_dataframe()


# In[161]:


rmses = np.array([])
articleIds = df['ArtikelID'].unique()
results_df = pd.DataFrame(columns=['ArtikelID', 'Skipped', 'Error', 'RMSE', 'TrainData Mean', 'aic', 'Known Params',
                                   'p','d','q','P','D','Q','S',
                                  ])
for articleId in articleIds:
    dfTemporary = df[(df['ArtikelID'].isin([articleId]))]
    dfTemporary = dfTemporary.drop(columns=['ArtikelID'])
    dfTemporary = group_by_frequence(dfTemporary, frequence='MS',startdate=start, enddate=end)
    dfTemporary = dfTemporary.drop(columns=['Datum'])
    y = dfTemporary['Menge']
    y_train = y[:date(2016,12,31)]
    X_train = y_train.index
    y_test = y[date(2017,1,1):date(2017,12,31)]
    X_test = y_test.index
    res = get_best_model(y_train)
    model = res['model']
    if model is None:
        display('Skipped %s' % articleId)
        results_df = results_df.append({'ArtikelID': articleId, 
                                        'Skipped': True,
                                        'Error': False},
                                       ignore_index=True)
        continue
    try:
        fitted_model = model.fit()
        #pred = fitted_model.get_prediction(dynamic=False,
        #                                    steps=12,
        #                                   start=pd.to_datetime('2016-12-01'),
        #                                   end=pd.to_datetime('2017-12-31'))
        pred = fitted_model.forecast(dynamic=True, steps=12)
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
    #plt.figure(figsize=(20,5))
    #display(pred)
    #display(y_test)
    #plt.plot(pred, color='red')
    #plt.plot(y_test, color='blue')
    #plt.show()
    #rmses = np.append(rmses, [get_rmse(y_test, pred.predicted_mean) / y_train.mean()])
    #print('RMSE; %s' % get_rmse(y_test, pred))
    #print('Mean; %s' % y_train.mean())
    results = {'ArtikelID': articleId, 
                                    'Skipped': False,
                                    'Error': False, 
                                    'RMSE': get_rmse(y_test, pred),
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
    rmses = np.append(rmses, [get_rmse(y_test, pred) / y_train.mean()])
    # display('RMSE: %s - Mean: %s' % (get_rmse(y_test, pred), y_train.mean()))
results_df = results_df.set_index('ArtikelID')
results_df['RMSE / TrainData Mean'] = results_df['RMSE'] / results_df['TrainData Mean']
#display('RMSE Mean: %s' % (np.mean(rmses)))


# In[162]:


import datetime
display(results_df['RMSE'].mean())
display(results_df)
outputfile = 'SARIMA_Pipeline_%s.csv' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
results_df.to_csv(path_or_buf='outputs/%s' % outputfile,
                     sep=';')


# In[135]:


df = pd.read_csv(filepath_or_buffer='outputs/SARIMA_Pipeline_2018-10-01_22-50-32.csv',
                 sep=';',
                 header=0)
# display(df)
display(df[(df['RMSE / TrainData Mean'] > 0.3) & (df['RMSE / TrainData Mean'] < 2)])


# In[164]:


results_df[results_df['RMSE']<1]


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

