
# coding: utf-8

# In[1]:


# coding: utf-8

# %matplotlib notebook
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
from sklearn.linear_model import LinearRegression


from datetime import date
enddate = date(2018, 1, 1)
startdate = date(2010, 1, 1)

def group_by_frequence(df, frequence='W'):
    df = df.copy()
    df = df.groupby([pd.Grouper(key='Datum', freq=frequence)])['Menge'].sum().reset_index()
    idx = pd.date_range(start=startdate, end=enddate, freq=frequence)
    df.index = pd.DatetimeIndex(df.Datum)
    df = df.reindex(idx, fill_value=0)
    return df

def get_dataframe_SingleArt(filename, frequence='W'):
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Single/'+filename+'.csv',
                     sep=';',
                     header=0,
                     usecols=[0,1])
    
    df['Datum'] = pd.to_datetime(df['Datum'], yearfirst=True, errors='raise')

     # convert FaktDatum to datetime
    # df.index = pd.DatetimeIndex(df.Datum, yearfirst=True)
    return group_by_frequence(df, frequence)

df = get_dataframe_SingleArt('felco2')
df = get_dataframe_SingleArt('bambus1201012')
df = df.drop(columns=['Datum'])
display(df.head())


# Die ersten 5 Datenreihen

# In[2]:


import datetime as dt
y = df['Menge']
splitdate = date(2016, 12, 25)
y_train = y[:splitdate]
y_test = y[splitdate:]
y_test = y_test[1:]
X_test = y_test.index.map(dt.datetime.toordinal).values.reshape(-1, 1)

plt.figure(figsize=(20,5))
plt.plot(y_train.index, y_train)

X_train = y_train.index.map(dt.datetime.toordinal).values.reshape(-1, 1)
y_train = y_train
regr = LinearRegression()
regr.fit(X_train, y_train)
prediction = regr.predict(X_train)
plt.plot(X_train, prediction)


years = [startdate, date(2015, 1, 1), date(2016, 1, 1), date(2017, 1, 1)]
for year in years:
    plt.axvline(x=year, color='black')
plt.show()


# Hier sieht man eine Übersicht über die verwertenden Daten. Die schwarzen Linien bezeichnen den Jahreswechsel und die orange eine lineare Regression über die Daten.

# In[3]:


mean = y_train.mean()
mean1 = y_train[:51].mean()
mean2 = y_train[52:104].mean()
mean3 = y_train[104:].mean()
means = [mean1, mean2, mean3]

plt.plot([1, 2, 3], means)
plt.plot([1, 2, 3], [mean, mean, mean])
plt.yticks(np.arange(0, max(means) * 1.10, max(means) / 5))
plt.title('Mittelwerte')
plt.show()

var = y_train.var()
var1 = y_train[:51].var()
var2 = y_train[52:104].var()
var3 = y_train[104:].var()
vars = [var1, var2, var3]
plt.plot([1, 2, 3], vars)
plt.plot([1, 2, 3], [var, var, var])
plt.yticks(np.arange(0, max(vars) * 1.10, max(vars) / 5))
plt.title('Varianz')
plt.show()


# In[4]:


fuller = sm.tsa.stattools.adfuller(y_train)
display(fuller)
adf_val = fuller[0]
p = fuller[1]
critical_values = fuller[4]

if adf_val < critical_values['5%']:
    print('Is stationary with p of %s' % p)
    d = 0
else:
    print('Is not stationary. Try differencing.')
    # TODO

# y_train_diff= y_train.diff(periods=1).values[1:]
# display(y_train_diff)
# plt.plot(y_train_diff)
# display(sm.tsa.stattools.adfuller(y_train_diff))
# y_train = y_train_diff


# In[5]:


import math
acf = sm.tsa.stattools.acf(y_train, nlags=150)
pacf = sm.tsa.stattools.pacf(y_train, nlags=150)
rangeacf = range(len(acf))
rangepacf = range(len(pacf))
plt.figure(figsize=(20,5))

plt.axhline(y=-1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.axhline(y=1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.plot(rangeacf, [1.96/math.sqrt((len(y_train)-x)) for x in rangeacf], color='green')
plt.plot(rangeacf, [-1.96/math.sqrt((len(y_train)-x)) for x in rangeacf], color='green')
plt.scatter(rangeacf, acf)
plt.title('ACF')
plt.show()
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(y_train, lags=150, ax=ax1)

# plt.plot(range(len(acf)), )
plt.figure(figsize=(20,5))
plt.scatter(rangepacf, pacf)
plt.axhline(y=-1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.axhline(y=1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.plot(rangeacf, [1.96/math.sqrt((len(y_train)-x)) for x in rangeacf], color='green')
plt.plot(rangeacf, [-1.96/math.sqrt((len(y_train)-x)) for x in rangeacf], color='green')
plt.title('PACF')
plt.show()
fig = plt.figure(figsize=(20,5))
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_acf(y_train, lags=150, ax=ax2)

over_line_acf = []
for i,x in enumerate(acf):
    if (x > 1.96/np.sqrt(len(y_train)) or -1.96/np.sqrt(len(y_train)) > x) and i > 0:
        over_line_acf.append(i)
over_line_pacf = []
for i,x in enumerate(pacf):
    if (x > 1.96/np.sqrt(len(y_train)) or -1.96/np.sqrt(len(y_train)) > x) and i > 0:
        over_line_pacf.append(i)
print(over_line_acf)
print(over_line_pacf)
print([x for x in over_line_acf if x in over_line_pacf])


# Die obigen beiden Plots zeigen, dass sowohl ACF wie auch PACF nach dem ersten Element stark abfallend sind. Dies deutet darauf hin, dass ein ARMA-Modell geeignet ist. Vorbedingung für die Korrektheit von ACF und PACF ist, dass die Daten stationär sind.

# In[6]:


p = 0
d = 0
q = 0
P = 1
D = 0
Q = 1
S = 52
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


# In[77]:


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

# In[7]:


import datetime as dt
df = get_dataframe_SingleArt('bambus1201012', 'MS')
df = df.drop(columns=['Datum'])
display(df.head())

y = df['Menge']
splitdate = date(2016, 12, 1)
y_train = y[:splitdate]
y_test = y[splitdate:]
y_test = y_test[1:]
X_test = y_test.index.map(dt.datetime.toordinal).values.reshape(-1, 1)

plt.figure(figsize=(20,5))
plt.plot(y_train.index, y_train)

X_train = y_train.index.map(dt.datetime.toordinal).values.reshape(-1, 1)
y_train = y_train
regr = LinearRegression()
regr.fit(X_train, y_train)
prediction = regr.predict(X_train)
plt.plot(X_train, prediction)


years = [startdate, date(2015, 1, 1), date(2016, 1, 1), date(2017, 1, 1)]
for year in years:
    plt.axvline(x=year, color='black')
plt.show()


# In[8]:


fuller = sm.tsa.stattools.adfuller(y_train)
display(fuller)
adf_val = fuller[0]
p = fuller[1]
critical_values = fuller[4]

if adf_val < critical_values['5%']:
    print('Is stationary with p of %s' % p)
    d = 0
else:
    print('Is not stationary. Try differencing.')
    # TODO

# y_train_diff= y_train.diff(periods=1).values[1:]
# display(y_train_diff)
# plt.plot(y_train_diff)
# display(sm.tsa.stattools.adfuller(y_train_diff))
# y_train = y_train_diff


# In[9]:


import math
acf = sm.tsa.stattools.acf(y_train, nlags=30)
pacf = sm.tsa.stattools.pacf(y_train, nlags=30)
rangeacf = range(len(acf))
rangepacf = range(len(pacf))
plt.figure(figsize=(20,5))

plt.axhline(y=-1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.axhline(y=1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.plot(rangeacf, [1.96/math.sqrt((len(y_train)-x)) for x in rangeacf], color='green')
plt.plot(rangeacf, [-1.96/math.sqrt((len(y_train)-x)) for x in rangeacf], color='green')
plt.scatter(rangeacf, acf)
plt.title('ACF')
plt.show()
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(y_train, lags=30, ax=ax1)

# plt.plot(range(len(acf)), )
plt.figure(figsize=(20,5))
plt.scatter(rangepacf, pacf)
plt.axhline(y=-1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.axhline(y=1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.plot(rangeacf, [1.96/math.sqrt((len(y_train)-x)) for x in rangeacf], color='green')
plt.plot(rangeacf, [-1.96/math.sqrt((len(y_train)-x)) for x in rangeacf], color='green')
plt.title('PACF')
plt.show()
fig = plt.figure(figsize=(20,5))
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_acf(y_train, lags=30, ax=ax2)

over_line_acf = []
for i,x in enumerate(acf):
    if (x > 1.96/np.sqrt(len(y_train)-i) or -1.96/np.sqrt(len(y_train) -i) > x) and i > 0:
        over_line_acf.append(i)
over_line_pacf = []
for i,x in enumerate(pacf):
    if (x > 1.96/np.sqrt(len(y_train) -i) or -1.96/np.sqrt(len(y_train) -i) > x) and i > 0:
        over_line_pacf.append(i)
print(over_line_acf)
print(over_line_pacf)
print([x for x in over_line_acf if x in over_line_pacf])


# In[24]:


p = 0
d = 1
q = 0
P = 1
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


# In[11]:


import itertools
import sys
import time
start = time.time()
# define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)
# generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
best_rmse = 10000000000000000000
best_aic = 10000000000000000000
best_seasonal_pdq = (-1,-1,-1)
best_seasonal_pdq_rmse = (-1,-1,-1)
best_shift = -5
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
print("Best RMSE SARIMAX{}x{} model - RMSE:{} with shift {}".format((0,0,0), best_seasonal_pdq_rmse, best_rmse, best_shift))


# In[31]:


# coding: utf-8

# %matplotlib notebook
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
from sklearn.linear_model import LinearRegression


from datetime import date
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
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Single/'+filename+'.csv',
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


# In[43]:


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


# In[45]:


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

