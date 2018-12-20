
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


# In[83]:


from datetime import date
enddate = date(2018, 1, 1)
startdate = date(2013, 1, 1)

def group_by_month(df):
    df = df.copy()
    df = df.groupby([pd.Grouper(key='Datum', freq='W')])['Menge'].sum().reset_index()
    idx = pd.date_range(start=df.Datum.min(), end=enddate, freq='W')
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
    return group_by_month(df)

# get_dataframe_SingleArt('felco2')


# In[84]:


def runStationary(data):
    #Fit the model 
    mod = sm.tsa.statespace.SARIMAX(df['Menge'], trend='c', order=(1,0,0)) 
    res = mod.fit(disp=False) 
    print(res.summary())

    mod = sm.tsa.statespace.SARIMAX(df['Menge'], trend='c', order=(2,0,0)) 
    res = mod.fit(disp=False) 
    print(res.summary())

    mod = sm.tsa.statespace.SARIMAX(df['Menge'], trend='c', order=(0,0,1)) 
    res = mod.fit(disp=False) 
    print(res.summary())

    mod = sm.tsa.statespace.SARIMAX(df['Menge'], trend='c', order=(1,0,1)) 
    res = mod.fit() 
    print(res.summary())


# In[85]:


def runARIMA_test(data):
    #Fit the model 
   # mod = ARIMA(df['Menge'].astype(np.float64), order=(1,2,1)) 
   # res = mod.fit(disp=0) 
   # print(res.summary())
    p=0
    q=0
    d=0
    pdq=[]
    aic=[]
    
    for p in range(6):
        for d in range(2):
            for q in range(4):
                try:
                    arima_mod=sm.tsa.ARIMA(df['Menge'].astype(np.float64),(p,d,q)).fit(transparams=True)

                    x=arima_mod.aic

                    x1= p,d,q
                    print (x1,x)

                    aic.append(x)
                    pdq.append(x1)
                except Exception as e:
                    pass
                    #print(e)
    keys = pdq
    values = aic
    d = dict(zip(keys, values))
    print (d)

    minaic=min(d, key=d.get)

    for i in range(3):
        p=minaic[0]
        d=minaic[1]
        q=minaic[2]
    print (p,d,q)


# In[114]:


from sklearn.linear_model import LinearRegression

df = get_dataframe_SingleArt('felco2')
adfstat = sm.tsa.stattools.adfuller(df['Menge'])
print('felco2: %s -> %s' % (adfstat[0], adfstat[2]))
regr = LinearRegression()
df2 = df.Datum.reshape(len(df.Datum), 1)
# regr.fit(df2,df.Menge)
plt.figure(figsize=(20,10))
plt.plot(df.Datum, df.Menge)
# plt.plot(df.Datum, regr.predict(df2))
plt.show()

df = get_dataframe_SingleArt('aquatexsg201')
adfstat= sm.tsa.stattools.adfuller(df['Menge'])
print('aquatexsg201: %s -> %s' % (adfstat[0], adfstat[2]))
regr = LinearRegression()
# regr.fit(df['Datum'], df['Menge'])
plt.figure(figsize=(20,10))
plt.plot(df.Datum, df.Menge)
# plt.plot(df.Datum, regr.predict(df['Datum']))
plt.show()

df = get_dataframe_SingleArt('bambus1201012')
adfstat= sm.tsa.stattools.adfuller(df['Menge'])
print('bambus1201012: %s -> %s' % (adfstat[0], adfstat[2]))
regr = LinearRegression()
# regr.fit(df['Datum'], df['Menge'])
plt.figure(figsize=(20,10))
plt.plot(df.Datum, df.Menge)
# plt.plot(df.Datum, regr.predict(df['Datum']))
plt.show()

df = get_dataframe_SingleArt('pfahloz25062_even')
adfstat= sm.tsa.stattools.adfuller(df['Menge'])
print('pfahloz25062_even: %s -> %s' % (adfstat[0], adfstat[2]))
regr = LinearRegression()
# regr.fit(df['Datum'], df['Menge'])
plt.figure(figsize=(20,10))
# plt.plot(df.Datum, regr.predict(df['Datum']))
plt.plot(df.Datum, df.Menge)
plt.show()


# In[32]:


df = get_dataframe_SingleArt('aquatexsg201')

#runStationary(df)
#runARIMA_test(df)
#Fit the model 
#mod = sm.tsa.statespace.SARIMAX(df['Menge'], trend='c', order=(1,2,1)) 
#res = mod.fit(disp=False) 
#print(res.summary())


# In[33]:


df = get_dataframe_SingleArt('bambus1201012')
# display(df['Menge'].astype(np.float64))


model = sm.tsa.ARIMA(df['Menge'].astype(np.float64), order=(5,1,1))
model_fit = model.fit()
print(model_fit.summary())

# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# In[ ]:


df = get_dataframe_SingleArt('bambus1201012')

runStationary(df)

# Fit the model
#mod = sm.tsa.statespace.SARIMAX(df['Menge'], trend='c', order=(1,1,1))
#res = mod.fit(disp=False)
#
#print(res.summary())


# In[ ]:


# Load the statsmodels api
import statsmodels.api as sm

# Load your dataset
endog = pd.read_csv('Datenexporte/Single/bambus1201012.csv', delimiter=';')

endog.index = pd.DatetimeIndex(endog.Datum)
endog['Menge']    = endog['Menge'].astype(float)
endog['ln_menge'] = np.log(endog['Menge'])
#endog['Datum'] = pd.to_datetime(df['Datum'], errors='raise')
    


# We could fit an AR(2) model, described above
mod_ar2 = sm.tsa.SARIMAX(endog['Menge'], order=(2,0,0))
# Note that mod_ar2 is an instance of the SARIMAX class

# Fit the model via maximum likelihood
res_ar2 = mod_ar2.fit()
# Note that res_ar2 is an instance of the SARIMAXResults class

# Show the summary of results
print(res_ar2.summary())

# We could also fit a more complicated model with seasonal components.
# As an example, here is an SARIMA(1,1,1) x (0,1,1,4):
mod_sarimax = sm.tsa.SARIMAX(endog['Menge'], order=(1,1,1),
                             seasonal_order=(0,1,1,2))
res_sarimax = mod_sarimax.fit()

# Show the summary of results
print(res_sarimax.summary())

