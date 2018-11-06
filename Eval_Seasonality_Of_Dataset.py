
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import date
from statsmodels.tsa.stattools import adfuller
import math
import itertools
import matplotlib.patches as mpatches
import statistics
import datetime

from statsmodels.tsa.seasonal import seasonal_decompose

default_startdate = date(2010, 1, 1)
default_enddate = date(2018, 1, 1)


# In[11]:


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
    df = pd.read_csv(filepath_or_buffer='../Datenexporte/Datenexport_200818/export_august.txt',
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

# perform Augmented Dickey Fuller test
# params: training set y, current value for d (count of differencing)
# returns Tupel of stationary data and d
def dickey_fuller(data, d):
    dftest = adfuller(data, autolag='AIC')
    adf_val = dftest[0]

    if adf_val < dftest[4]['5%']:
        # is stationary
        return (data, d)
    else:
        #not stationary -> differencing.
        data = np.diff(data)
        d = d+1
        data, d = dickey_fuller (data, d)
        if (d >= 3):
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
    
    seasonal = [round(j) for i, j in enumerate(result.seasonal)]
    peak = max(seasonal)
    res = [i for i, j in enumerate(seasonal) if j == peak]

    S = 0
    if len(res)>1 :
        S=res[1]-res[0]
        
    return S
    
    


# In[12]:


# import the data

import datetime as dt
start = date(2010, 1, 1)
end = date(2018, 1, 1)
# define data to calc for (or empty for all)
my_id =[722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830]

# load data
df = get_dataframe(my_id)
print('done import')


# In[14]:


# continue after above code executed:
# get seasonal for all, evaluate distribution

results_df = pd.DataFrame(columns=['ArtikelID', 'd', 'S'])
artId = df['ArtikelID'].unique()
skip = 0
for aid in artId:
# for each artId, get data, group monthly, discard unneeded cols
    dfTemporary = df[(df['ArtikelID'].isin([aid]))]
    dfTemporary = dfTemporary.drop(columns=['ArtikelID'])
    dfTemporary = group_by_frequence(dfTemporary, frequence='MS',startdate=start, enddate=end)
    dfTemporary = dfTemporary.drop(columns=['Datum'])
    
    # keep only amounts per month
    y = dfTemporary['Menge']
    
    # need diff?
    try:
        data, d = dickey_fuller(y, 0)
        # get ACF
        S = getSeasonality(y)
        results = {'ArtikelID': aid, 'd': d, 'S': S}
        results_df = results_df.append(results, ignore_index=True)
    except:
        skip = skip + 1
        # print ('skipped: %i' % aid)

results_df = results_df.set_index('ArtikelID')

#print results_df['S'].value_counts()
counts = results_df['S'].value_counts().to_dict()
print (counts)
print ('skipped: %i' % skip)
print ('total Art.: %i' % len(artId))
# all => {12: 2815, 1: 694, 2: 75, 3: 49, 4: 34, 5: 30, 7: 26, 8: 21, 6: 18, 9: 14, 10: 5, 11: 4}

# print(results_df)

