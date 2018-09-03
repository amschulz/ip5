
# coding: utf-8

# In[369]:


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
startdate = date(2014, 1, 1)

def group_by_week(df):
    df = df.copy()
    df = df.groupby([pd.Grouper(key='Datum', freq='W')])['Menge'].sum().reset_index()
    idx = pd.date_range(start=startdate, end=enddate, freq='W')
    df.index = pd.DatetimeIndex(df.Datum)
    df = df.reindex(idx, fill_value=0)
    return df

def get_dataframe_SingleArt(filename):
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Single/'+filename+'.csv',
                     sep=';',
                     header=0,
                     usecols=[0,1])
    
    df['Datum'] = pd.to_datetime(df['Datum'], yearfirst=True, errors='raise')

     # convert FaktDatum to datetime
    # df.index = pd.DatetimeIndex(df.Datum, yearfirst=True)
    return group_by_week(df)

df = get_dataframe_SingleArt('felco2')
df = get_dataframe_SingleArt('bambus1201012')
df = df.drop(columns=['Datum'])
display(df.head())


# Die ersten 5 Datenreihen

# In[370]:


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

# In[371]:


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


# In[372]:


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


# In[373]:


import math
acf = sm.tsa.stattools.acf(y_train, nlags=100)
pacf = sm.tsa.stattools.pacf(y_train, nlags=100)
rangeacf = range(len(acf))
rangepacf = range(len(pacf))
# plt.plot(rangeacf, [1.96/math.sqrt((len(acf)-x)) for x in rangeacf])
plt.figure(figsize=(20,5))

plt.axhline(y=-1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.axhline(y=1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.scatter(rangeacf, acf)
plt.title('ACF')
plt.show()

# plt.plot(range(len(acf)), )
plt.figure(figsize=(20,5))
plt.scatter(rangepacf, pacf)
plt.axhline(y=-1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.axhline(y=1.96/np.sqrt(len(y_train)), linestyle='--', color='grey')
plt.title('PACF')
plt.show()


# Die obigen beiden Plots zeigen, dass sowohl ACF wie auch PACF nach dem ersten Element stark abfallend sind. Dies deutet darauf hin, dass ein ARMA-Modell geeignet ist. Vorbedingung für die Korrektheit von ACF und PACF ist, dass die Daten stationär sind.

# In[381]:


p = 0
d = 0
q = 0
P = 2
D = 0
Q = 0
S = 52
param = (p, d, q)
param_seasonal = (P, D, Q, S)

model = sm.tsa.statespace.SARIMAX(y_train,
                                  order = param,
                                  seasonal_order = param_seasonal,
                                  enforce_stationarity=True,
                                  enforce_invertibility=True)
forecast = model.fit(disp=False).forecast(len(X_test))

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

