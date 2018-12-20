
# coding: utf-8

# In[2]:


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


# In[11]:


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

output_file = 'results.csv'
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
df = get_dataframe()
# df = get_dataframe([722, 3986, 7979, 7612, 239, 1060, 5841, 6383, 5830])
# df = get_dataframe([6383])
ids = df['ArtikelID'].unique()

amount_total = len(ids)
end1 = time.time()
print('Duration 1 in seconds: %s' % (end1 - start))

skipped = 0
amount = 0
correct = 0
false = 0
rest_correct = 0

output_f = open(output_file, 'w')
csv_writer = csv.writer(output_f, delimiter=';', lineterminator='\n')
csv_writer.writerow(['ArtikelID', 'Kategorie Jahre einzeln', 'Kategorie Durchschnitt', 'IsSame'])

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
    
    csv_writer.writerow([i, category_years, category_mean, category_mean == category_years])
    
    if category_years != category_mean:
        false = false + 1
    else:
        correct =  correct + 1
        if category_years == 'Gruppe: 6 - Rest' or category_years == 'Gruppe: 5 - Rest':
            rest_correct = rest_correct + 1

output_f.close()
            
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

