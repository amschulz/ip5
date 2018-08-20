
# coding: utf-8

# In[3]:


# %matplotlib notebook
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def get_phases(df):
    columns = ['LinReg - Echt %', 'LinReg - Echt % MovAvg', 'LinReg - Echt % 1 Digit']
    phases = {}
    for column in columns:
        phases_col = []
        for row in df['LinReg - Echt % MovAvg']:
            if math.isnan(row):
                continue
            name = ''
            if row < 0.2 and row > -0.2:
                name = 'Neutral'
            elif row <= -0.2:
                name = 'Negativ'
            else:
                name = 'Positiv'
            if not phases_col:
                phases_col.append({'name': name, 'length': 1})
            else:
                if phases_col[-1]['name'] == name:
                    phases_col[-1]['length'] = phases_col[-1]['length'] + 1
                else:
                    phases_col.append({'name': name, 'length': 1})
        phases_notneutral = []
        phases_negative = []
        phases_positive = []
        phases_neutral = []
        for x in phases_col:
            if x['name'] != 'Neutral':
                phases_notneutral.append(x)
        for x in phases_col:
            if x['name'] == 'Negativ':
                phases_negative.append(x)
        for x in phases_col:
            if x['name'] == 'Positiv':
                phases_positive.append(x)
        for x in phases_col:
            if x['name'] == 'Neutral':
                phases_neutral.append(x)
        phases[column] = {
            'notneutral': phases_notneutral,
            'negative': phases_negative,
            'positive': phases_positive,
            'neutral': phases_neutral
        }
    return phases

def get_peaks(df):
    df['Gehört zu Peak'] = (df['LinReg - Echt % 1 Digit'] < 0) & (df['VerAvg - Echt % 1 Digit'] < 0)
    display(df)
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
    return peaks

def is_linear(df, phases, peaks):
    return len(peaks) == 0
    
    
    '''
    print(phases['LinReg - Echt % MovAvg']['negative'])
    print(phases['LinReg - Echt % MovAvg']['positive'])
    return len (phases['LinReg - Echt % MovAvg']['negative']) == 0 and len(phases['LinReg - Echt % MovAvg']['positive']) == 0
    '''    
    
    '''
    g1_linear = False
    for row in df['LinReg - Echt % MovAvg']:
        if not math.isnan(row) and row != 0:
            print('linear: orange not 0')
            break
    else:
        max01 = False
        for i in df['LinReg - Echt % 1 Digit']:
            if i == 0.1 or i == -0.1:
                if not max01:
                    max01 = True
                else:
                    print('linear: already seen 0.1')
                    break
            if i > 0.1 or i < -0.1:
                print('linear: value out of 0.1')
                break
        else:
            g1_linear = True
    return g1_linear'''
   
        
def is_onelongpeak(df, phases, peaks):
    real_peaks = []
    for peak in peaks:
        print(peak)
        if peak['length'] > 1:
            real_peaks.append(peak)
    print('rela_peaks')
    print(real_peaks)
    if len(real_peaks) == 1:
        print(real_peaks[0])
        return real_peaks[0]['length'] >= 3
    else:
        return False 
    
    '''
    if len(phases['LinReg - Echt % 1 Digit']['negative']) != 1:
        return False
    return phases['LinReg - Echt % 1 Digit']['negative'][0]['length'] > 2
    '''
    '''
    if len(phases['LinReg - Echt % MovAvg']['negative']) != 1:
        return False
    if phases['LinReg - Echt % MovAvg']['negative'][0]['length'] < 2:
        return False
    
    alreadyfound = False
    for el in phases['LinReg - Echt % 1 Digit']['negative']:
        if el['length'] >= 2:
            if alreadyfound:
                return False
            else:
                alreadyfound = True
    else:
        return alreadyfound
    '''

def is_oneshortpeak(df, phases, peaks):
    return len(peaks) == 1 and peaks[0]['length'] < 3
    
    '''
    if len(phases['LinReg - Echt % 1 Digit']['negative']) != 1:
        return False
    return phases['LinReg - Echt % 1 Digit']['negative'][0]['length'] <= 2
    '''
    '''
    if len(phases['LinReg - Echt % MovAvg']['negative']) != 1:
        return False
    if phases['LinReg - Echt % MovAvg']['negative'][0]['length'] > 2:
        return False
    
    minimum = 0
    amount = 0
    for month in df['LinReg - Echt % MovAvg']:
        if month >= 0:
            continue
        else:
            if minimum > month:
                minimum = month
                amount = 0
            elif minimum == month:
                amount += 1
    return amount == 1
    '''            

def is_twolongpeaks(df, phases, peaks):
    real_peaks = []
    for peak in peaks:
        if peak['length'] > 1:
            real_peaks.append(peak)
    if len(real_peaks) == 2:
        return real_peaks[0]['length'] >= 3 and real_peaks[1]['length'] >= 3
    else:
        return False 
    '''
    if len(phases['LinReg - Echt % 1 Digit']['negative']) != 2:
        return False
    peak1 = phases['LinReg - Echt % 1 Digit']['negative'][0]['length'] < 2
    peak2 = phases['LinReg - Echt % 1 Digit']['negative'][1]['length'] < 2
    return peak1 or peak2
    '''

def is_twoshortpeaks(df, phases, peaks):
    return len(peaks) == 2 and peaks[0]['length'] < 3 and peaks[1]['length'] < 3

    '''
    if len(phases['LinReg - Echt % 1 Digit']['negative']) != 2:
        return False
    peak1 = phases['LinReg - Echt % 1 Digit']['negative'][0]['length'] >= 2
    peak2 = phases['LinReg - Echt % 1 Digit']['negative'][1]['length'] >= 2
    return peak1 or peak2
    '''

def categorize(df):
    printit = True
    peaks = get_peaks(df)
    phases = get_phases(df)
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

def display_results(ds):
    for el in ds:
        typ = el['typ']
        d = el['d']
        df1 = pd.DataFrame(data=d)
        regr = LinearRegression()
        X_train = df1['Monat']
        y_train = df1['Verkaufsmenge']
        X_train = X_train.values.reshape(12, 1)
        y_train = y_train.values.reshape(12, 1)
        regr.fit(X_train, y_train)
        df1['Lineare Regression'] = regr.predict(X_train)
        # df1['Gleitende Durchschnitte'] = df1.rolling(window=3).mean()['Verkaufsmenge']
        # df1['LinReg - GleDur'] = (df1['Lineare Regression'] - df1['Gleitende Durchschnitte'])
        # df1['Betrag LinReg - GleDur'] = (df1['LinReg - GleDur']) * (df1['LinReg - GleDur'])
        # df1['Betrag LinReg - GleDur']= df1['Betrag LinReg - GleDur'].apply(np.sqrt)
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(1, 2 ,1)
        df1['Verkaufsmenge Avg'] = df1['Verkaufsmenge'].mean()
        df1['VerAvg - Echt'] = df1['Verkaufsmenge Avg'] - df1['Verkaufsmenge']
        df1['LinReg - Echt'] = (df1['Lineare Regression'] - df1['Verkaufsmenge'])
        
        
        # df1['LinReg - Echt %'] = df1['LinReg - Echt'].divide(df1['Verkaufsmenge'])
        df1['LinReg - Echt %'] = df1['LinReg - Echt'] /((df1['Verkaufsmenge']) + 0.001)
        df1['VerAvg - Echt %'] = df1['VerAvg - Echt'] /((df1['Verkaufsmenge']) + 0.001)
        
        
        df1['LinReg - Echt % 1 Digit'] = df1['LinReg - Echt %'].round(1)
        df1['VerAvg - Echt % 1 Digit'] = df1['VerAvg - Echt %'].round(1)

        df1['LinReg - Echt % MovAvg'] = df1.rolling(window=3).mean()['LinReg - Echt %'].round(1)
        # df1['LinReg - Echt % 1 Digit MovAvg'] = df1.rolling(window=3).mean()['LinReg - Echt % 1 Digit'].round(1)


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
        # plt8 = plt.plot(X_train, df1['LinReg - Echt % 1 Digit MovAvg'], color='#316200')
        


        #plt.yticks(np.arange(-1, 1.1, 0.1))
        plt.yticks(np.arange(-2, 1, 0.3))
        plt.legend(handles=[mpatches.Patch(color='black', label='LinReg - Echt %'),
                            mpatches.Patch(color='#d500d5', label='LinReg - Echt % 1 Digit'),
                            mpatches.Patch(color='#FF9650', label='LinReg - Echt % MovAvg'),
                            # mpatches.Patch(color='#316200', label='LinReg - Echt % 1 Digit MoAvg'),
                            mpatches.Patch(color='blue', label='Wert 0')])
        # fig.suptitle('Trendbestimmung von Artikel %s (%s) ' % (product_id, muster))
        plt.show()
        # display(df1)
        display(categorize(df1))


# In[60]:


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


# In[61]:


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


# In[62]:


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


# In[63]:


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


# In[64]:


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


# In[65]:


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


# In[66]:


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


# In[67]:


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


# In[68]:


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


# In[69]:


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


# In[4]:


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


def group_by_month(df):
    df = df.copy()
    df = df.groupby(['ArtikelID', pd.Grouper(key='FaktDatum', freq='MS')])['Menge'].sum().reset_index()
    idx = pd.date_range(start=df.FaktDatum.min(), end=df.FaktDatum.max(), freq='MS')
    display(df)
    df.index = pd.DatetimeIndex(df.FaktDatum)
    df = df.reindex(idx, fill_value=0)
    df = df.drop(columns=['FaktDatum', 'ArtikelID'])
    df['Date'] = df.index.map(dt.datetime.toordinal)
    return {'df': df,
           'listoflabels': []}

def group_by_month1(df):
    df = df.copy()
    df = df.groupby(['ArtikelID', pd.Grouper(key='FaktDatum', freq='MS')])['Menge'].sum().reset_index()
    idx = pd.date_range(start=df.FaktDatum.min(), end=df.FaktDatum.max(), freq='MS')
    df.index = pd.DatetimeIndex(df.FaktDatum)
    df = df.reindex(idx, fill_value=0)
    df = df.drop(columns=['FaktDatum', 'ArtikelID'])
    df['Date'] = df.index.map(dt.datetime.toordinal)
    return {'df': df,
           'listoflabels': []}

def get_dataframe():
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_050518/Verkaufszahlen_v2.txt',
                     sep=';',
                     header=0,
                     usecols=[0,6,8,10])
    df = df.rename(index=str, columns={'fk_ArtikelBasis_ID_l':'ArtikelID'})
        
    # remove inactive products
    df = df[df['Inaktiv_b'] == 'Falsch']
    df = df[(df['ArtikelID'].isin([722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830]))]

     # convert FaktDatum to datetime
    df['FaktDatum'] = pd.to_datetime(df['FaktDatum'], errors='coerce')
    
    # remove old datasets
    delta = date.today() - timedelta(365 * 5)
    df = df[(df['FaktDatum'] >= delta) & (df['FaktDatum'] < date(2017, 10, 1))]
    return df
    
df = get_dataframe()
ids = df['ArtikelID'].unique()

for i in ids:
    df2 = df.copy()
    df2 = df2[(df2['ArtikelID'] == i)]
    df2 = group_by_month1(df2)['df']
    df2['Date'] = df2.index
    df2['Monat'] = df2.Date.dt.month
    df2 = df2.groupby(['Monat'])['Menge'].mean().reset_index()
    df2['Verkaufsmenge'] = df2['Menge']
    ds = [{'typ': '%s' % i,
            'd': df2},]
    display_results(ds)
    break

