
# coding: utf-8

# In[4]:


# %matplotlib notebook
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def get_peaks(df):
    df['Gehört zu Peak'] = (df['LinReg - Echt % 1 Digit'] < 0) & (df['VerAvg - Echt % 1 Digit'] < 0)
    verlauf = []
    for num, row in enumerate(df['Gehört zu Peak']):
        if not verlauf:
            verlauf.append({'IsPeak': row, 'from': 1, 'to': 1, 'length': 1})
        else:
            last = verlauf[-1]
            if last['IsPeak'] == row:
                last['to'] = num + 1
                last['length'] = last['length'] + 1
                verlauf[-1] = last
            else:
                verlauf.append({'IsPeak': row, 'from': num + 1, 'to': num + 1, 'length': 1})
    peaks = []
    for phase in verlauf:
        if phase['IsPeak']:
            peaks.append(phase)
    
    if peaks and peaks[-1]['to'] == 12 and peaks[0]['from'] == 1:
        lastpeak = peaks[-1]
        firstpeak = peaks[0]
        lastpeak['to'] = firstpeak['to']
        lastpeak['length'] = lastpeak['length'] + peaks[0]['length'] 
        peaks[-1] =  lastpeak
        del peaks[0]
            
    return peaks

def is_linear(df, phases, peaks):
    return len(peaks) == 0
   
        
def is_onelongpeak(df, phases, peaks):
    real_peaks = []
    for peak in peaks:
        if peak['length'] > 1:
            real_peaks.append(peak)
    if len(real_peaks) == 1:
        return real_peaks[0]['length'] >= 3
    else:
        return False 

def is_oneshortpeak(df, phases, peaks):
    return len(peaks) == 1 and peaks[0]['length'] < 3         

def is_twolongpeaks(df, phases, peaks):
    real_peaks = []
    for peak in peaks:
        if peak['length'] > 1:
            real_peaks.append(peak)
    if len(real_peaks) == 2:
        return real_peaks[0]['length'] >= 3 and real_peaks[1]['length'] >= 3
    else:
        return False 

def is_twoshortpeaks(df, phases, peaks):
    return len(peaks) == 2 and peaks[0]['length'] < 3 and peaks[1]['length'] < 3

def get_group_of_peaks(peaks):
    real_peaks = []
    for peak in peaks:
        if peak['length'] > 1:
            real_peaks.append(peak)
    l = len(real_peaks)
    if l == 0:
        return 'Gruppe: 1 - Linear'
    elif l == 1:
        return 'Gruppe: 2 - 1 Peak'
    elif l == 2:
        return 'Gruppe: 3 - 2 Peaks'
    elif l == 3:
        return 'Gruppe: 4 - 3 Peaks '
    else:
        return 'Gruppe: 6 - Rest'
    
def categorize(df):
    printit = False
    peaks = get_peaks(df)
    phases = []
    # phases = get_phases(df)
    # return get_group_of_peaks(peaks)
    linear = is_linear(df, phases, peaks)
    onelongpeak = is_onelongpeak(df, phases, peaks)
    oneshortpeak = is_oneshortpeak(df, phases, peaks)
    twolongpeaks = is_twolongpeaks(df, phases, peaks)
    twoshortpeaks = is_twoshortpeaks(df, phases, peaks)
    
    if printit:
        print('----- Start of printing')
        # print('Phasen: %s' % phases)
        if linear:
            print('Gruppe: 1 - Linear')
        if onelongpeak:
            print('Gruppe: 2 - 1 lange Hoch')
        if oneshortpeak:
            print('Gruppe: 3 - 1 kurzes Hoch')
        if twolongpeaks:
            print('Gruppe: 4 - 2 lange Hochs')
        if twoshortpeaks:
            print('Gruppe: 5 - 2 kurze Hochs')
        print('----- End of printing')
    
    if linear:
        return 'Gruppe: 1 - Linear'
    if onelongpeak:
        return 'Gruppe: 2 - 1 lange Hoch'
    if oneshortpeak:
        return 'Gruppe: 3 - 1 kurzes Hoch'
    if twolongpeaks:
        return 'Gruppe: 4 - 2 lange Hochs'
    if twoshortpeaks:
        return 'Gruppe: 5 - 2 kurze Hochs'
    return 'Gruppe: 6 - Rest'

def display_results(ds, show_prints=False):
    categories = {}
    for el in ds:
        typ = el['typ']
        d = el['d']
        df1 = pd.DataFrame(data=d)
        regr = LinearRegression()
        X_train = df1['Monat']
        y_train = df1['Verkaufsmenge']
        try:
            X_train = X_train.values.reshape(12, 1)
            y_train = y_train.values.reshape(12, 1)
        except ValueError as e:
            display(el['typ'])
            display(df1)
            raise ValueError
            
        regr.fit(X_train, y_train)
        df1['Lineare Regression'] = regr.predict(X_train)
        df1['Verkaufsmenge Avg'] = df1['Verkaufsmenge'].mean()
        df1['VerAvg - Echt'] = df1['Verkaufsmenge Avg'] - df1['Verkaufsmenge']
        df1['LinReg - Echt'] = (df1['Lineare Regression'] - df1['Verkaufsmenge'])
        df1['LinReg - Echt %'] = df1['LinReg - Echt'] /((df1['Verkaufsmenge']) + 0.001)
        df1['VerAvg - Echt %'] = df1['VerAvg - Echt'] /((df1['Verkaufsmenge']) + 0.001)
        df1['LinReg - Echt % 1 Digit'] = df1['LinReg - Echt %'].round(1)
        df1['VerAvg - Echt % 1 Digit'] = df1['VerAvg - Echt %'].round(1)
        df1['LinReg - Echt % MovAvg'] = df1.rolling(window=3).mean()['LinReg - Echt %'].round(1)        
        if show_prints: 
            fig = plt.figure(figsize=(20, 5))
            plt.subplot(1, 2 ,1)
            plt1 = plt.scatter(X_train, y_train,  color='black', label='Tatsächliche Werte')
            plt2 = plt.plot(X_train, regr.predict(X_train), color='blue', linewidth=3)
            plt3 = plt.plot(X_train, df1.rolling(window=3).mean()['Verkaufsmenge'], color='red')
            plt9 = plt.plot(X_train, df1['Verkaufsmenge Avg'], color='#316200')
            plt.yticks(np.arange(0, df1['Verkaufsmenge'].max(), 10))
            plt.legend(handles=[plt1,
                                mpatches.Patch(color='blue', label='Lineare regression'),
                                mpatches.Patch(color='red', label='Moving average'),
                                mpatches.Patch(color='#316200', label='Verkaufsmenge Avg'),
                               ])
            plt.title('Trendbestimmung %s' % typ)
            plt.subplot(1, 2 ,2)
            plt4 = plt.plot(X_train, df1['LinReg - Echt %'], color='black', label='LinReg - Echt %')
            plt5 = plt.scatter(X_train, [0,0,0,0,0,0,0,0,0,0,0,0], color='blue', label='Wert 0')
            plt6 = plt.plot(X_train, df1['LinReg - Echt % 1 Digit'], color='#d500d5', label='LinReg - Echt % 1 Digit')
            plt7 = plt.plot(X_train, df1['LinReg - Echt % MovAvg'], color='#FF9650')

            plt.yticks(np.arange(-2, 1, 0.3))
            plt.legend(handles=[mpatches.Patch(color='black', label='LinReg - Echt %'),
                                mpatches.Patch(color='#d500d5', label='LinReg - Echt % 1 Digit'),
                                mpatches.Patch(color='#FF9650', label='LinReg - Echt % MovAvg'),
                                # mpatches.Patch(color='#316200', label='LinReg - Echt % 1 Digit MoAvg'),
                                mpatches.Patch(color='blue', label='Wert 0')])

            plt.show()
        # display(df1)
        category = categorize(df1)
        categories[el['typ']] = category
    return categories


# In[5]:


ds1 = [{'typ': 'linear steigend',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,110,110,120,125,130,130,140,145,150,150]}},
       {'typ': 'linear fallend',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,95,90,80,80,75,70,65,60,60,50]}},
       {'typ': 'linear neutral',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,95,100,100,95,105,100,105,100,100,100]}}
      ]
ds = ds1
display_results(ds)


# In[6]:


ds2 = [{'typ': '1 langes Hoch 1',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,120,120,130,120,100,100,95,100,105,110,95]}},
       {'typ': '1 langes Hoch 2',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,95,100,130,130,140,120,95,100,105,100,95]}},
       {'typ': '1 langes Hoch 3',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,105,100,95,100,100,120,130,125,120,95]}},
      ]
ds = ds2
display_results(ds)


# In[7]:


ds3 = [{'typ': '1 kurzes Hoch 1',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,120,130,105,100,100,95,100,100,105,100,100]}},
       {'typ': '1 kurzes Hoch 2',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,95,105,100,100,95,100,100,130,120,100]}},
       {'typ': '1 kurzes Hoch 3',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,95,100,105,100,130,120,100,100,105,100,100]}},
      ]
ds = ds3
display_results(ds)


# In[8]:


ds4 = [{'typ': '2 lange Hochs 1',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,120,120,130,120,100,100,110,130,120,120,100]}},
       {'typ': '2 lange Hochs 2',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [120,120,130,100,95,100,100,110,130,120,120,100]}},
       {'typ': '2 lange Hochs 3',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100, 100,120,120,130,120,100,110,130,120,120,100]}},
       
      ]
ds = ds4
display_results(ds)


# In[9]:


ds5 = [{'typ': '2 kurze Hochs 1',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [100,120,130,105,100,95,95,105,100,130,120,100]}},
       {'typ': '2 kurze Hochs 2',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [100,100,95,120,130,95,120,130,100,95,95,100]}},
       {'typ': '2 kurze Hochs 3',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [100,95,100,105,100,120,130,105,100,130,120,100]}},
      ]
ds = ds5
display_results(ds)


# In[10]:


ds6 = [{'typ': '1 Hoch, Lang + Zeroes 1',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [100,120,130,105,100,0,0,0,0,0,0,0]}},
       {'typ': '1 Hoch, Lang + Zeroes 2',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [0,0,0,0,130,100,120,130,0,0,0,0]}},
       {'typ': '1 Hoch, Lang + Zeroes 3',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [0,0,0,0,0,0,0,0,100,130,100,120]}},
      ]
ds = ds6
display_results(ds)


# In[11]:


ds7 = [{'typ': 'felco 4 16',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [3,0,2,4,1,4,0,0,0,0,7,1]}},
       {'typ': 'felco 4 17',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [7,7,4,2,0,6,1,6,3,2,2,0]}},
       
       {'typ': 'felco 2 16',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [74,32,77,40,35,34,77,118,87,31,70,72]}},
       
       {'typ': 'rechen500k. 2+ peaks',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [40,7,65,80,22,43,73,44,41,50,153,79]}},
      ]
ds = ds7
display_results(ds)


# In[12]:


ds8 = [{'typ': '2 hoche Hochs 1',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [10,90,130,120,80,10,0,100,130,150,100,50]}},
       {'typ': '2 hoche Hochs 2',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [120,80,10,0,100,130,150,100,50,10,90,130]}},
       {'typ': '2 tiefe Hochs 1',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [10,25,25,20,9,0,7,15,20,20,15,10]}},
       {'typ': '2 tiefe Hochs 2',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [25,20,9,0,7,15,20,20,15,10,10,25]}},
      ]
ds = ds8
display_results(ds)


# In[13]:


ds9 = [{'typ': '1 langes Hochs 1',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [90,120,580,700,690,600,350,110,100,0,30,100]}},
       {'typ': '1 langes Hochs 2',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [10,9,0,200,270,310,280,230,40,80,90,10]}},
       {'typ': '1 langes Hochs 3',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [210, 250, 200, 90, 10, 0, 0, 10, 50, 100, 105, 110]}},
      ]
ds = ds9
display_results(ds)


# In[14]:


ds10 = [{'typ': '2 kurze Hochs 1',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [10,120,130,10,5,0,0,6,0,80,100,20]}},
       {'typ': '2 kurze Hochs 2',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [530,90,10,5,0,3,45,50,450,340,90,220]}},
       {'typ': '2 kurze, flache Hochs mit Null',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [26,19,17,0,10,27,26,17,15,16,16,20]}},
       {'typ': '2 kurze, flache Hochs mit 0',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [10,60,90,0,9,5,7,50,90,10,15,10]}},
      ]
ds = ds10
display_results(ds)


# In[30]:


import csv
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime
from datetime import date
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import datetime as dt
import time

enddate = date(2017, 1, 1)
yearstowatch = 4

def group_by_month1(df):
    df = df.copy()
    df = df.groupby(['ArtikelID', pd.Grouper(key='FaktDatum', freq='MS')])['Menge'].sum().reset_index()
    idx = pd.date_range(start=df.FaktDatum.min(), end=enddate, freq='MS')
    df.index = pd.DatetimeIndex(df.FaktDatum)
    df = df.reindex(idx, fill_value=0)
    df = df.drop(columns=['FaktDatum', 'ArtikelID'])
    df['Date'] = df.index.map(dt.datetime.toordinal)
    return {'df': df,
           'listoflabels': []}

def get_dataframe(ids_to_track=[]):
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_200818/export_august.txt',
                     sep=';',
                     header=0,
                     usecols=[0,6,8,10])
    df = df.rename(index=str, columns={'fk_ArtikelBasis_ID_l':'ArtikelID'})

    if ids_to_track:
        df = df[(df['ArtikelID'].isin(ids_to_track))]

     # convert FaktDatum to datetime
    df['FaktDatum'] = pd.to_datetime(df['FaktDatum'], dayfirst=True, errors='raise')
    
    # remove old datasets
    delta = enddate - timedelta(365 * yearstowatch)
    df = df[(df['FaktDatum'] >= delta) & (df['FaktDatum'] < enddate)]
    return df

def get_categories_for_years(df2):
    for year in df2.Jahr.unique():
        df3 = df2.copy()
        df3['Verkaufsmenge'] = df3['Menge']
        df3 = df3[df3['Jahr'] == year]
        df3 = df3.drop(columns=['Date'])
        if df3.Monat.max() == 12 and df3.Monat.min() == 1:
            fullyears.append({'typ': '%s %s' % (i, year), 'd': df3})
        else:
            continue
    if  len(fullyears) < 3:
        return {}
    
    categories_yearly = display_results(fullyears, False)
    
    sumofvals = 0
    val_sum = {}
    for val in categories_yearly.values():
        sumofvals = sumofvals + 1
        if val not in val_sum.keys():
            val_sum[val] = 1
        else:
            val_sum[val] = val_sum[val] + 1
    if sumofvals < 3:
        return {'sumofvals': sumofvals, 'val_sum': {}}
    else:
        return {'sumofvals': sumofvals, 'val_sum': val_sum}
    
def extract_main_category_from_years(categories_yearly):
    group = 'Gruppe: 6 - Rest'
    for key, val in categories_yearly['val_sum'].items():
        if (val / categories_yearly['sumofvals']) > 0.5:
            group = key
            break
    return group

def get_category_for_mean(df2):
    # df4 = df2.copy()
    df2 = df2.groupby(['Monat'])['Menge'].mean().reset_index()
    # df4['Verkaufsmenge'] = df4['Menge']
    # df4 = df4.drop(columns=['Menge'])
    # df4 = df4.merge(df2[['Monat', 'Menge']],
    #            on=['Monat'],
    #            how='outer')
    #display(df4)
    # df4['Streuung'] = df4['']
    df2['Verkaufsmenge'] = df2['Menge']
    ds = [{'typ': 'Mean',
            'd': df2}]
    result = display_results(ds, False)
    return result['Mean']

start = time.time()
#df = get_dataframe()
# df = get_dataframe([722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830])
df = get_dataframe([6383])
ids = df['ArtikelID'].unique()

amount_total = len(ids)
end1 = time.time()
print('Duration 1 in seconds: %s' % (end1 - start))

skipped = 0
amount = 0
correct = 0
false = 0
rest_correct = 0

for i in ids:
    df2 = df.copy()
    df2 = df2[(df2['ArtikelID'] == i)]
    df2 = group_by_month1(df2)['df']
    df2['Date'] = df2.index
    df2['Jahr'] = df2.Date.dt.year
    df2['Monat'] = df2.Date.dt.month
    fullyears = []
    categories_yearly = get_categories_for_years(df2)
    if not categories_yearly or not categories_yearly['val_sum']:
        skipped = skipped + 1
        continue
    category_years = extract_main_category_from_years(categories_yearly)
    category_mean = get_category_for_mean(df2)
    amount = amount + 1
    
    if category_years != category_mean:
        false = false + 1
    else:
        correct =  correct + 1
        if category_years == 'Gruppe: 6 - Rest' or category_years == 'Gruppe: 5 - Rest':
            rest_correct = rest_correct + 1

end2 = time.time()
print('Duration 2 in seconds: %s' % (end2 - end1))
            
assert correct + false == amount        
print('Scope: %s years' % yearstowatch)
print('Amount of products: %s' % amount_total)
print('Skipped: %s' % skipped)
print('Amount: %s' % amount)
print('Correct: %s' % correct)
print('    in rest: %s' % rest_correct)
print('False: %s' % false)
print('Correctnes: %s' % (correct/amount))

