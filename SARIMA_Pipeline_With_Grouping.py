
# coding: utf-8

# In[239]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import math
import itertools
import matplotlib.patches as mpatches
import statistics

import datetime as dt
from datetime import datetime
from datetime import date

from scipy import signal

default_startdate = date(2010, 1, 1)
default_enddate = date(2018, 1, 1)

start = date(2010, 1, 1)
end = date(2018, 1, 1)

path = '../Datenexporte/Datenexport_200818/export_august.txt'


# In[275]:


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

# loads data from export file to DataFrame, selectable by ids
def get_dataframe(ids_to_track=[]):
    df = pd.read_csv(filepath_or_buffer=path,
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
    

    
    
    

# calculates ACF, PACF and differenced ACF, PACF and plots the charts
# parameter: series, data to work with
#            lags, number of lags to calculate
def show_acf_pacf(series, lags):
    import statsmodels.tsa.api as smt
    # 1x diff
    y_diff1= np.diff(series)

    # setup layout
    fig = plt.figure(figsize=(14, 5))
    layout = (2, 2)  # rows, cols
    acf_ax = plt.subplot2grid(layout, (0, 0))
    pacf_ax = plt.subplot2grid(layout, (0, 1))
    acfd_ax = plt.subplot2grid(layout, (1, 0))
    pacfd_ax = plt.subplot2grid(layout, (1, 1))

    # acf and pacf
    acfd = sm.tsa.stattools.acf(series, nlags=lags)
    smt.graphics.plot_acf(series, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(series, lags=lags, ax=pacf_ax, alpha=0.5) 

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

    
# display Results from Sarima Model
# returns Model, to be used for prediction
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
    S = getSeasonality(y_train)
    D = getSeasonalDifferencing(y_train, 0, S)
    xx, d = dickey_fuller(y_train, 0)
    
    # id d + D > 2 then set d = 0
    if (d+D >2): d=0
    known_params['d'] = d
    known_params['D'] = D
    known_params['S'] = S
    
    # evaluate p, q (ARIMA Part)
    
    # get acf-data and pacf-data. The critical borders are also part of the return value!
    acf  = sm.tsa.stattools.acf (y_train_stationary, nlags=36, alpha=0.5)
    pacf = sm.tsa.stattools.pacf(y_train_stationary, nlags=36, alpha=0.5)
    
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
    display(acf_peaks)
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
    results = eval_pdq_by_example(maxparam=4, y_train=y_train, **known_params)
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


# In[276]:


# load one, examine ACF and PACF 

# sample ids = [722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830]
# define time range:
start = date(2010, 1, 1)
end = date(2018, 1, 1)

#define id to calculate (single examination)
artId = [5151]

# load and group data to monthly
frame = get_dataframe(artId)
grouped = group_by_frequence(frame)


# prepare data for training
series = grouped['Menge']

# make train and test set
splitdate = date(2016, 12, 31)
y_train = series[:splitdate]
y_test = series[splitdate:]
y_test = y_test[1:]
X_test = y_test.index.map(dt.datetime.toordinal).values.reshape(-1, 1)

#print(series)

# output results

show_acf_pacf(y_train, 30)
#diff = np.diff(y_train)
#show_acf_pacf(diff, 48)


# In[ ]:


# show prediction -> issue with date, needs fixing
def predictModel (mFit, data):

    # in-sample-prediction and confidence bounds
    # prediction time range
    pdate_start = pd.to_datetime('2017-01-01', yearfirst=True, errors='raise')
    pdate_end   = pd.to_datetime('2018-01-01', yearfirst=True, errors='raise')
    chart_start = pd.to_datetime('2010-01-01', yearfirst=True, errors='raise')
    
    pred = mFit.get_prediction(start=pdate_start, 
                              end=pdate_end)
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
    


# In[277]:


# Example code to evaluate a single art for differencing:


# load and group data to monthly
aid = 5151
g = group_by_frequence(get_dataframe([aid]))
data = g.drop(columns=['Datum'])

series = data['Menge']

y_train = series[:date(2016,12,31)]
X_train = y_train.index
y_test = y[date(2017,1,1):date(2017,12,31)]
X_test = y_test.index

diffData, d = dickey_fuller(y_train, 0)
D = getSeasonalDifferencing(y_train,12)
print ('Differencing Result for Art. %s: d= %i, D= %i' % (aid, d, D))


# In[273]:


# import the complete data set
artId = []

# define data to calc for (or empty for all)
artId = [722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830]

# load from export file
df = get_dataframe(artId)

print('done import')


# In[274]:


# continue after above code executed:
# get seasonal for all, evaluate distribution

results_df = pd.DataFrame(columns=['ArtikelID', 'd', 'D', 'S'])

artId = df['ArtikelID'].unique()
skip = 0
for aid in artId:
# for each artId, get data, group monthly and discard unneeded cols
    dfTemporary = df[(df['ArtikelID'].isin([aid]))]
    dfTemporary = dfTemporary.drop(columns=['ArtikelID'])
    dfTemporary = group_by_frequence(dfTemporary, frequence='MS',startdate=start, enddate=end)
    dfTemporary = dfTemporary.drop(columns=['Datum'])
    
    # keep only amounts per month
    series = dfTemporary['Menge']

    # make train and test set
    y_train = series[:date(2016,12,31)]
    X_train = y_train.index
    y_test = y[date(2017,1,1):date(2017,12,31)]
    X_test = y_test.index
    # need diff?
    try:
        data, d = dickey_fuller(y_train, 0)
        
        S = getSeasonality(y_train)
        D = getSeasonalDifferencing(y_train, 0, S)
        
        # append to results
        results = {'ArtikelID': aid, 'd': d, 'D': D, 'S': S}
        results_df = results_df.append(results, ignore_index=True)
    except:
        skip = skip + 1
        # print ('skipped: %i' % aid)

results_df = results_df.set_index('ArtikelID')

counts = results_df['D'].value_counts().to_dict()
print (counts)
print ('skipped: %i' % skip)
print ('total Art.: %i' % len(artId))
# counts for S =  {12: 3602, 1: 91, 2: 20, 3: 17, 7: 12, 6: 10, 8: 8, 4: 7, 5: 6, 10: 5, 9: 4, 11: 3}
# skipped: 120
# total Art.: 3905
# print(results_df)


# In[ ]:


rmses = np.array([])
articleIds = df['ArtikelID'].unique()
results_df = pd.DataFrame(columns=['ArtikelID', 'Skipped', 'Error', 'Mape', 
                                   'Median', 'TrainData Mean', 'aic', 'Known Params',
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
    # display('RMSE: %s - Mean: %s' % (get_rmse(y_test, pred), y_train.mean()))
results_df = results_df.set_index('ArtikelID')
#display('RMSE Mean: %s' % (np.mean(rmses)))


# In[7]:


print('Mape-Mean: %s' % results_df['Mape'].mean())
print('Median-Mean: %s' % results_df['Median'].mean())
display(results_df)
outputfile = 'SARIMA_Pipeline_%s.csv' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
results_df.to_csv(path_or_buf='outputs/%s' % outputfile, sep=';')


# In[5]:


id = 722

print('p: %s' % results_df['p'][id])
print('d: %s' % results_df['d'][id])
print('q: %s' % results_df['q'][id])
print('P: %s' % results_df['P'][id])
print('D: %s' % results_df['D'][id])
print('Q: %s' % results_df['Q'][id])
print('S: %s' % results_df['S'][id])


# In[279]:


df_read = pd.read_csv(filepath_or_buffer='../Datenexporte/SARIMA_Pipeline_2018-10-30_17-21-15.csv',
                 sep=';',
                 header=0)
#df_read = results_df
#df_read['ArtikelID'] = df_read.index
display(df_read)
# display(df[(df['RMSE / TrainData Mean'] > 0.3) & (df['RMSE / TrainData Mean'] < 2)])


# In[280]:


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

show_article_details(5151)


