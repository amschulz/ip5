
# coding: utf-8

# In[3]:


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
            if row == 0:
                name = 'Neutral'
            elif row < 0:
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
            if x['name'] != 'Positiv':
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

def is_oneshortpeak(df, phases):
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
                

def is_twolongpeaks(df, phases):
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
        df1['LinReg - Echt %'] = df1['LinReg - Echt'] / df1['Verkaufsmenge']
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
        # plt4 = plt.plot(X_train, df1['LinReg - Echt %'], color='black', label='LinReg - Echt %')
        plt5 = plt.scatter(X_train, [0,0,0,0,0,0,0,0,0,0,0,0], color='blue', label='Wert 0')
        plt6 = plt.plot(X_train, df1['LinReg - Echt % 1 Digit'], color='#d500d5', label='LinReg - Echt % 1 Digit')
        plt7 = plt.plot(X_train, df1['LinReg - Echt % MovAvg'], color='#FF9650')
        # plt8 = plt.plot(X_train, df1['LinReg - Echt % 1 Digit MovAvg'], color='#316200')


        #plt.yticks(np.arange(-1, 1.1, 0.1))
        plt.yticks(np.arange(-0.3, 0.31, 0.1))
        plt.legend(handles=[mpatches.Patch(color='black', label='LinReg - Echt %'),
                            mpatches.Patch(color='#d500d5', label='LinReg - Echt % 1 Digit'),
                            mpatches.Patch(color='#FF9650', label='LinReg - Echt % MovAvg'),
                            mpatches.Patch(color='#316200', label='LinReg - Echt % 1 Digit MoAvg'),
                            mpatches.Patch(color='blue', label='Wert 0')])
        # fig.suptitle('Trendbestimmung von Artikel %s (%s) ' % (product_id, muster))
        plt.show()
        # display(df1)
        display(categorize(df1))


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[36]:


def display_results2(ds, title):
    handles = []
    for i, el in enumerate(ds):
        typ = el['typ']
        d = el['d']
        color = el['color']
        df1 = pd.DataFrame(data=d)
        regr = LinearRegression()
        X_train = df1['Monat']
        y_train = df1['Verkaufsmenge']
        X_train = X_train.values.reshape(12, 1)
        y_train = y_train.values.reshape(12, 1)

        #fig = plt.figure(figsize=(20, 5))
        #plt.subplot(1, i+1, 1)

        plt.plot(X_train, y_train,  color=color)
        handles.append(mpatches.Patch(color=color, label=typ))
    plt.yticks(np.arange(0, 150, 10))
    plt.xticks(np.arange(1, 13, 1))
    plt.ylabel('Verkaufsmenge')
    plt.xlabel('Monate des Jahres')
    plt.legend(handles=handles)
    plt.title(title)
    # plt.subplot(1, 2 ,2)
    plt.show()
ds1 = [{'typ': 'linear steigend',
        'color': 'black',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,110,110,120,125,130,130,140,145,150,150]}},
       {'typ': 'linear fallend',
        'color': 'orange',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,95,90,80,80,75,70,65,60,60,50]}},
       {'typ': 'linear stabil',
        'color': 'blue',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,95,100,100,95,105,100,105,100,100,100]}}
      ]
ds = ds1
display_results2(ds, 'Lineare Gruppe')


# In[39]:


ds2 = [{'typ': '1 langer Peak am Jahresanfang',
        'color': 'black',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,120,120,130,120,100,100,95,100,105,110,95]}},
       {'typ': '1 langer Peak in der Jahresmitte',
        'color': 'orange',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,95,100,130,130,140,120,95,100,105,100,95]}},
       {'typ': '1 langer Peak am Jahresende',
        'color': 'blue',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,105,100,95,100,100,120,130,125,120,95]}},
      ]
ds = ds2
display_results2(ds, 'Gruppe mit einem langen Peak')


# In[40]:


ds3 = [{'typ': '1 kurzer Peak am Jahresanfang',
        'color': 'black',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,120,130,105,100,100,95,100,100,105,100,100]}},
       {'typ': '1 kurzer Peak in der Jahresmitte',
        'color': 'orange',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,100,95,105,100,100,95,100,100,130,120,100]}},
       {'typ': '1 kurzer Peak am Jahresende',
        'color': 'blue',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,95,100,105,100,130,120,100,100,105,100,100]}},
      ]
ds = ds3
display_results2(ds, 'Gruppe mit einem kurzen Peak')


# In[44]:


ds4 = [{'typ': '2 lange Peaks im Frühling und Herbst',
        'color': 'black',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100,120,120,130,120,100,100,110,130,120,130,100]}},
       {'typ': '2 lange Peaks im Winter und Herbst',
        'color': 'orange',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [120,120,130,100,95,100,100,110,130,130,120,100]}},
       {'typ': '2 lange Peaks im späten Frühling und Herbst',
        'color': 'blue',
        'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              'Verkaufsmenge': [100, 100,120,120,130,120,100,110,130,120,120,100]}},
       
      ]
ds = ds4
display_results2(ds, 'Zwei lange Peaks')


# In[45]:


ds5 = [{'typ': '2 kurze Peaks im Winter und Spätherbst',
        'color': 'black',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [100,120,130,105,100,95,95,105,100,130,120,100]}},
       {'typ': '2 kurze Peaks im Frühling und Spätsommer',
        'color': 'orange',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [100,100,95,120,130,95,120,130,100,95,95,100]}},
       {'typ': '2 kurze Peaks im Sommer und Spätherbst',
        'color': 'blue',
       'd': {'Monat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             'Verkaufsmenge': [100,95,100,105,100,120,130,105,100,130,120,100]}},
      ]
ds = ds5
display_results2(ds, 'Zwei kurze Peaks')

