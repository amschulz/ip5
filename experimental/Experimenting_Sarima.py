
# coding: utf-8

# In[208]:


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


# In[ ]:


def get_dataframe():
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_200818/export_august.txt',
                     sep=';',
                     header=0,
                     usecols=[0,6,8,10])
    df = df.rename(index=str, columns={'fk_ArtikelBasis_ID_l':'ArtikelID'})
    df = df[(df['ArtikelID'].isin([6383]))]

     # convert FaktDatum to datetime
    df['FaktDatum'] = pd.to_datetime(df['FaktDatum'], dayfirst=True, errors='raise')
    return df


# In[93]:


def get_dataframe_SingleArt(filename):
    df = pd.read_csv(filepath_or_buffer='../Datenexporte/Single/'+filename+'.csv',
                     sep=';',
                     header=0,
                     usecols=[0,1])
    
     # convert FaktDatum to datetime
    df['ln_menge'] = np.log(df['Menge'])    
    df.index = pd.DatetimeIndex(df.Datum)
    return df


# In[118]:


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


# In[179]:


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


# In[205]:


df = get_dataframe_SingleArt('felco2')
adfstat, pvalue, critvalues, resstore = sm.tsa.stattools.adfuller(df['Menge'], maxlag=1, regression='ct', autolag='AIC', store=True, regresults=False)
print('felco2:', adfstat)

df = get_dataframe_SingleArt('aquatexsg201')
adfstat, pvalue, critvalues, resstore = sm.tsa.stattools.adfuller(df['Menge'], maxlag=1, regression='ct', autolag='AIC', store=True, regresults=False)
print('aquatexsg201:', adfstat)

df = get_dataframe_SingleArt('bambus1201012')
adfstat, pvalue, critvalues, resstore = sm.tsa.stattools.adfuller(df['Menge'], maxlag=1, regression='ct', autolag='AIC', store=True, regresults=False)
print('bambus1201012:', adfstat)


# In[187]:


df = get_dataframe_SingleArt('aquatexsg201')

#runStationary(df)
#runARIMA_test(df)
#Fit the model 
#mod = sm.tsa.statespace.SARIMAX(df['Menge'], trend='c', order=(1,2,1)) 
#res = mod.fit(disp=False) 
#print(res.summary())


# In[221]:


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


# In[110]:


df = get_dataframe_SingleArt('bambus1201012')

runStationary(df)

# Fit the model
#mod = sm.tsa.statespace.SARIMAX(df['Menge'], trend='c', order=(1,1,1))
#res = mod.fit(disp=False)
#
#print(res.summary())


# In[80]:


# Load the statsmodels api
import statsmodels.api as sm

# Load your dataset
endog = pd.read_csv('../Datenexporte/Single/bambus1201012.csv', delimiter=';')

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


# In[ ]:



df = get_dataframe()
ids = df['ArtikelID'].unique()

amount = 0
correct = 0
fals = 0
rest_correct = 0
pr = True

for i in ids:
    df2 = df.copy()
    df2 = df2[(df2['ArtikelID'] == i)]
    df2 = group_by_month1(df2)['df']
    df2['Date'] = df2.index
    df2['Jahr'] = df2.Date.dt.year
    df2['Monat'] = df2.Date.dt.month
    fullyears = []
    for year in df2.Jahr.unique():
        df3 = df2.copy()
        df3['Verkaufsmenge'] = df3['Menge']
        df3 = df3[df3['Jahr'] == year]
        df3 = df3.drop(columns=['Date'])
        if df3.Monat.max() == 12 and df3.Monat.min() == 1:
            fullyears.append({'typ': '%s %s' % (i, year), 'd': df3})
        else:
            continue
    if not fullyears:
        continue
    categories_yearly = display_results(fullyears)
 
    df2 = df2.groupby(['Monat'])['Menge'].mean().reset_index()
    df2['Verkaufsmenge'] = df2['Menge']
    ds = [{'typ': '%s mean' % i,
            'd': df2}]
    categories_mean = display_results(ds)
    
    sumofvals = 0
    val_sum = {}
    for val in categories_yearly.values():
        sumofvals = sumofvals + 1
        if val not in val_sum.keys():
            val_sum[val] = 1
        else:
            val_sum[val] = val_sum[val] + 1
    if sumofvals < 3:
        continue
    
    amount = amount + 1
    
    group = ''
    for key, val in val_sum.items():
        if (val / sumofvals) > 0.5:
            group = key
            break
    else:
        fals = fals + 1
        continue
    
    if group != categories_mean['%s mean' % i]:
        fals = fals + 1
    else:
        correct =  correct + 1
        if group == 'Gruppe: 6 - Rest' or group == 'Gruppe: 5 - Rest':
            rest_correct = rest_correct + 1

assert correct + fals == amount        

print('amount: %s' % amount)
print('correct: %s' % correct)
print('    davon rest: %s' % rest_correct)
print('false: %s' % fals)

print('correctness: %s' % (correct/amount))

