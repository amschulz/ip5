
# coding: utf-8

# In[136]:


#%matplotlib notebook
# coding: utf-8

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
#        for row in df['LinReg - Echt % MovAvg']:
        for row in df['LinReg - Echt %']:
            if math.isnan(row):
                continue
            name = ''
            if (row > -0.1) & (row < 0.1):
                name = 'Neutral'
            elif row < -0.1:
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
            if x['name'] == 'Negativ':
                phases_negative.append(x)
            if x['name'] == 'Positiv':
                phases_positive.append(x)
            if x['name'] == 'Neutral':
                phases_neutral.append(x)
        phases[column] = {
            'notneutral': phases_notneutral,
            'negative': phases_negative,
            'positive': phases_positive,
            'neutral': phases_neutral
        }
    return phases

def is_linear(df):
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
    return g1_linear
        
def is_onelongpeak(df, phases):
    
    display(phases['LinReg - Echt % 1 Digit']['negative'])
    
    if len(phases['LinReg - Echt % 1 Digit']['negative']) != 1:
        return False
    if phases['LinReg - Echt % 1 Digit']['negative'][0]['length'] <= 2:
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

def is_oneshortpeak(df, phases):
    if len(phases['LinReg - Echt % 1 Digit']['negative']) != 1:
        print("short, false1")
        return False
    if phases['LinReg - Echt % 1 Digit']['negative'][0]['length'] > 2:
        print("short, false2")
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
                

def is_twolongpeaks(df, phases):
    if len(phases['LinReg - Echt % 1 Digit']['negative']) > 1:
        print("two peaks found!")
    return  False

def is_twoshortpeaks(df, phases):
    return  False

def categorize(df):
    printit = True
    phases = get_phases(df)
    linear = is_linear(df)
    onelongpeak = is_onelongpeak(df, phases)
    oneshortpeak = is_oneshortpeak(df, phases)
    twolongpeaks = is_twolongpeaks(df, phases)
    twoshortpeaks = is_twoshortpeaks(df, phases)
    
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
        df1['LinReg - Echt'] = (df1['Lineare Regression'] - df1['Verkaufsmenge'])
        
       # df1['LinReg - Echt %'] = df1['LinReg - Echt'].divide(df1['Verkaufsmenge'])
        df1['LinReg - Echt %'] = df1['LinReg - Echt'] /((df1['Verkaufsmenge']) + 0.001)
        df1['LinReg - Echt % 1 Digit'] = df1['LinReg - Echt %'].round(1)
        
        df1['LinReg - Echt % MovAvg'] = df1.rolling(window=3).mean()['LinReg - Echt %'].round(1)
        df1['LinReg - Echt % 1 Digit MovAvg'] = df1.rolling(window=3).mean()['LinReg - Echt % 1 Digit'].round(1)


        plt1 = plt.scatter(X_train, y_train,  color='black', label='Tatsächliche Werte')
        plt2 = plt.plot(X_train, regr.predict(X_train), color='blue', linewidth=3)
        plt3 = plt.plot(X_train, df1.rolling(window=3).mean()['Verkaufsmenge'], color='red')
        plt.yticks(np.arange(0, df1['Verkaufsmenge'].max(), 10))
        plt.legend(handles=[plt1,
                            mpatches.Patch(color='blue', label='Lineare regression'),
                            mpatches.Patch(color='red', label='Moving average')])
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
                            mpatches.Patch(color='#316200', label='LinReg - Echt % 1 Digit MoAvg'),
                            mpatches.Patch(color='blue', label='Wert 0')])
        # fig.suptitle('Trendbestimmung von Artikel %s (%s) ' % (product_id, muster))
        plt.show()
        display(df1)
        display(categorize(df1))


# In[57]:


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
# display_results(ds)


# In[29]:


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
# display_results(ds)


# In[30]:


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
# display_results(ds)


# In[31]:


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
# display_results(ds)


# In[32]:


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
# display_results(ds)


# In[43]:


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
#display_results(ds)


# In[34]:


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
# display_results(ds)


# In[42]:


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
#display_results(ds)


# In[41]:


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
#display_results(ds)


# In[36]:


ds10 = [{'typ': '2 kurze Hochs 1',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [10,120,130,10,5,0,0,6,0,80,100,20]}},
       {'typ': '2 kurze Hochs 2',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [530,90,10,5,0,3,45,450,340,90, 50,220]}},
       {'typ': '2 kurze, flache Hochs mit Null',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [26,19,17,5,10,27,26,17,15,16,16,20]}},
       {'typ': '2 kurze, flache Hochs mit 0',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [10,60,90,5,9,5,7,50,90,10,15,10]}},
      ]
ds = ds10
# display_results(ds)


# In[137]:


# testdata zur erkennung von langen/kurzen Hochs am Anfang
# erkennung ist mässig, kurze hochs eher unmöglich

mnt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ds2 = [{'typ': '1 langes Hoch, deutlich',
        'd': {'Monat': mnt, 'Verkaufsmenge': [100,180,190,185,120,100,100,95,100,105,100,95]}},
       {'typ': '1 langes Hoch, flach',
        'd': {'Monat': mnt,  'Verkaufsmenge': [5,7,7,7,7,5,5,5,5,5,5,5]}},
       {'typ': '1 kurzes Hoch, deutlich',
        'd': {'Monat': mnt,  'Verkaufsmenge': [100,120,50,44,45,20,33,27,20,20,25,20]}},
       {'typ': '1 kurzes Hoch, flach',
        'd': {'Monat': mnt,  'Verkaufsmenge': [7,7,5,5,5,5,5,5,5,5,5,5]}},
#       {'typ': '2 lange Hoch',
#        'd': {'Monat': mnt,
#              'Verkaufsmenge': [100,150,165,155,95,100,100,100,195,200,190,95]}},
#       {'typ': '2 kurze Hoch',
#        'd': {'Monat': mnt,
#              'Verkaufsmenge': [100,100,165,120,95,100,100,90,150,145,90,95]}},
      ]
ds = ds2
display_results(ds)


# In[118]:


# testdata erkennen eines langen hoch am anfang
# versuch mit transformieren der daten
mnt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
d1 = [140,180,190,185,120,100,100,95,100,105,100,95]
d2 = [690,970,1040,540,20,3,10,2,4, 10,2,5]
d3 = [4,2,5,15,10,12,2,1,0,3,2,1]

def transform(data):
    dx = []
    mean = np.mean(data)
    div = 100.0/mean
    for v in data:
        if (v >= mean):
            value = (v-mean)*div
        else:
            value = -1*(v-mean)*div
        res = 0
        if (value < 10):     res = 0
        elif (value < 100):  res = 1
        elif (value < 200):  res = 2
        elif (value >= 200): res = 3    
        dx.append(res)
    return dx


ds = [{'typ': '1 langes Hoch am Anfang', 'd': {'Monat': mnt, 'Verkaufsmenge': transform(d1)}},
      {'typ': '1 langes Hoch extrem',    'd': {'Monat': mnt, 'Verkaufsmenge': transform(d2)}},
      {'typ': '1 langes Hoch flach',     'd': {'Monat': mnt, 'Verkaufsmenge': transform(d3)}},
     ]

#display_results(ds)

