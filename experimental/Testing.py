
# coding: utf-8

# In[3]:


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

def get_dataframe():
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_050518/Verkaufszahlen_v2.txt',
                     sep=';',
                     header=0,
                     usecols=[0,6,8,10])
    df = df.rename(index=str, columns={'fk_ArtikelBasis_ID_l':'ArtikelID'})
        
    # remove inactive products
    df = df[df['Inaktiv_b'] == 'Falsch']
    df = df[(df['ArtikelID'] == 3547)]

     # convert FaktDatum to datetime
    df['FaktDatum'] = pd.to_datetime(df['FaktDatum'], errors='coerce')
    
    # remove old datasets
    delta = date.today() - timedelta(365 * 3)
    df = df[(df['FaktDatum'] >= delta) & (df['FaktDatum'] < date(2017, 10, 1))]
    
    return df

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

def group_by_month2(df):
    df = df.copy()
    df = df.groupby(['ArtikelID', pd.Grouper(key='FaktDatum', freq='MS')])['Menge'].sum().reset_index()
    
    # create dummy product and append it to the dataframe.
    # This way we have all dates in the dataframe even if none of the products was sold.
    dummyproductsales = []
    date_range = pd.date_range(start=df.FaktDatum.min(), end=df.FaktDatum.max(), freq='MS')
    for week_start in date_range:
        dummyproductsales.append([-1, week_start, 0])
    dummydf = pd.DataFrame(dummyproductsales, columns=['ArtikelID', 'FaktDatum', 'Menge'])
    df = df.append(dummydf)
    
    # df['FaktDatum'] = df['FaktDatum'].dt.strftime('%Y-%m-%d')
    listoflabels = df['FaktDatum'].unique()
    df = df.pivot(index='ArtikelID', columns='FaktDatum', values='Menge')
    df = df.fillna(0)
    df = df.drop(-1)
    return {'df': df,
           'listoflabels': listoflabels}

orig_data = get_dataframe()
data_month = group_by_month1(orig_data)
df2 = data_month['df']
display(df2)


# In[4]:


from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches

regr = LinearRegression()
X_train = df2.Date
y_train = df2.Menge
X_train = X_train.values.reshape(-1, 1)
y_train = y_train.values
regr.fit(X_train, y_train)

prediction = regr.predict(X_train)

plt1 = plt.scatter(X_train, y_train,  color='black', label='Tatsächliche Werte')
plt2 = plt.plot(X_train, prediction, color='blue', linewidth=3)
plt3 = plt.plot(X_train, df2.rolling(window=3).mean()['Menge'], color='red')
plt.title('Trendbestimmung')
plt.legend(handles=[plt1,
                    mpatches.Patch(color='blue', label='Lineare regression'),
                    mpatches.Patch(color='red', label='Moving average')])
plt.show()


# In[5]:


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
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches

def get_dataframe(liste):
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_050518/Verkaufszahlen_v2.txt',
                     sep=';',
                     header=0,
                     usecols=[0,6,8,10])
    df = df.rename(index=str, columns={'fk_ArtikelBasis_ID_l':'ArtikelID'})
        
    # remove inactive products
    df = df[df['Inaktiv_b'] == 'Falsch']
    #df = df[(df['ArtikelID'].isin([3547, 6798, 722, 4073, 965]))]
    df = df[(df['ArtikelID'].isin(liste))]
    # convert FaktDatum to datetime
    df['FaktDatum'] = pd.to_datetime(df['FaktDatum'], errors='coerce')
    
    # remove old datasets
    delta = date.today() - timedelta(365 * 3)
    df = df[(df['FaktDatum'] >= date(2014, 1, 1)) & (df['FaktDatum'] < date(2017, 1, 1))]
    
    return df

def group_by_month1(df):
    retval = {}
    for product_id in df.ArtikelID.unique():
        df2 = df.copy()
        df2 = df2[(df['ArtikelID'] == product_id)]
        df2 = df2.groupby(['ArtikelID', pd.Grouper(key='FaktDatum', freq='MS')])['Menge'].sum().reset_index()
        idx = pd.date_range(start=df2.FaktDatum.min(), end=df2.FaktDatum.max(), freq='MS')
        df2.index = pd.DatetimeIndex(df2.FaktDatum)
        df2 = df2.reindex(idx, fill_value=0)
        df2 = df2.drop(columns=['FaktDatum', 'ArtikelID'])
        df2['Date'] = df2.index.map(dt.datetime.toordinal)
        retval[str(product_id)] = df2
    return retval

def trendbestimmung(product_id, df2, muster):
    df2_work = df2.copy()
    df2_work['Date_clear'] = df2_work.index
    df2_years = df2_work.copy()
    df2_years = df2_years.groupby([pd.Grouper(key='Date_clear', freq='YS')])['Menge'].sum().reset_index()
    years = df2_years['Date_clear'].unique()
    yearends = pd.date_range(start=df2_years.Date_clear.min(), end=df2_years.Date_clear.max() + np.timedelta64(1, 'Y'), freq='Y')
    fig = plt.figure(figsize=(20, 5))
    for i, year in enumerate(years):
        df2_year = df2_work[(df2_work['Date_clear'] >= year) & (df2_work['Date_clear'] < yearends[i])]
        df2_year = df2_year.drop(columns=['Date_clear'])
        regr = LinearRegression()
        X_train = df2_year.Date
        y_train = df2_year.Menge
        X_train = X_train.values.reshape(-1, 1)
        y_train = y_train.values
        regr.fit(X_train, y_train)

        prediction = regr.predict(X_train)
        plt.subplot(1, 3 ,i+1)
        ts = pd.to_datetime(str(year)) 
        d = ts.strftime('%Y')
        plt.title('Jahr %s'% (d))
        plt1 = plt.scatter(df2_year.index, y_train,  color='black', label='Tatsächliche Werte')
        plt2 = plt.plot(df2_year.index, prediction, color='blue', linewidth=3)
        plt3 = plt.plot(df2_year.index, df2_year.rolling(window=3).mean()['Menge'], color='red')
        plt.legend(handles=[plt1,
                        mpatches.Patch(color='blue', label='Linear regression'),
                        mpatches.Patch(color='red', label='Moving average')])
    fig.suptitle('Trendbestimmung von Artikel %s (%s) ' % (product_id, muster))
    plt.show()

    
hortima_muster = {
    'Typ1 - Linear':[722, 173],
    'Typ2 - 1 runder Peak':[4073],
    'Typ3 - 1 spitziger Peak':[],
    'Typ4 - 2 runde Peaks':[],
    'Typ5 - 2 spitzige Peaks':[965],
    'Typ6 - Klumpenrisiko(Rest)': [6798, 982]
}
artikel_liste = []
for a, b in hortima_muster.items():
    artikel_liste = artikel_liste + b
orig_data = get_dataframe(artikel_liste)
data_month = group_by_month1(orig_data)
df2 = data_month

muster = ''

for i,v in df2.items():
    int_i = int(i)
    for mu, liste in hortima_muster.items():
        if int_i in liste:
            muster = mu
    trendbestimmung(i, v, muster)


# In[6]:


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
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches

def get_dataframe(liste):
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_050518/Verkaufszahlen_v2.txt',
                     sep=';',
                     header=0,
                     usecols=[0,6,8,10])
    df = df.rename(index=str, columns={'fk_ArtikelBasis_ID_l':'ArtikelID'})
        
    # remove inactive products
    df = df[df['Inaktiv_b'] == 'Falsch']
    #df = df[(df['ArtikelID'].isin([3547, 6798, 722, 4073, 965]))]
    df = df[(df['ArtikelID'].isin(liste))]
    # convert FaktDatum to datetime
    df['FaktDatum'] = pd.to_datetime(df['FaktDatum'], errors='coerce')
    
    # remove old datasets
    delta = date.today() - timedelta(365 * 3)
    df = df[(df['FaktDatum'] >= date(2014, 1, 1)) & (df['FaktDatum'] < date(2017, 1, 1))]
    
    return df

def group_by_month1(df):
    retval = {}
    for product_id in df.ArtikelID.unique():
        df2 = df.copy()
        df2 = df2[(df['ArtikelID'] == product_id)]
        df2 = df2.groupby(['ArtikelID', pd.Grouper(key='FaktDatum', freq='MS')])['Menge'].sum().reset_index()
        idx = pd.date_range(start=df2.FaktDatum.min(), end=df2.FaktDatum.max(), freq='MS')
        df2.index = pd.DatetimeIndex(df2.FaktDatum)
        df2 = df2.reindex(idx, fill_value=0)
        df2 = df2.drop(columns=['FaktDatum', 'ArtikelID'])
        df2['Date'] = df2.index.map(dt.datetime.toordinal)
        retval[str(product_id)] = df2
    return retval

def trendbestimmung(product_id, df2, muster):
    df2_work = df2.copy()
    df2_work['Date_clear'] = df2_work.index
    df2_work['Month'] = df2_work.Date_clear.dt.month
    df2_work['Year'] = df2_work.Date_clear.dt.year
    df2_years = df2_work.copy()
    df2_years = df2_years.groupby([pd.Grouper(key='Date_clear', freq='YS')])['Menge'].sum().reset_index()
    years = df2_years['Date_clear'].unique()
    yearends = pd.date_range(start=df2_years.Date_clear.min(), end=df2_years.Date_clear.max() + np.timedelta64(1, 'Y'), freq='Y')
    fig = plt.figure(figsize=(20, 5))
    plts = []
    cols = {'2014': '#d500d5',
            '2015': '#62f731',
            '2016': '#316200'}
    for i, year in enumerate(years):
        df2_year = df2_work[(df2_work['Date_clear'] >= year) & (df2_work['Date_clear'] < yearends[i])]
        # display(df2_work.Date_clear.dt.year)
        df2_year = df2_year.drop(columns=['Date_clear'])
        y_train = df2_year.Menge
        y_train = y_train.values
        display(df2_year['Year'].unique()[0])
        plt1 = plt.scatter(df2_year['Month'], y_train,
                           color=cols[str(df2_year['Year'].unique()[0])],
                           label='Tatsächliche Werte (%s)' % df2_year['Year'].unique()[0])
        plts.append(plt1)

        # plt2 = plt.plot(df2_year['Month'], color='blue')
    plts.append(mpatches.Patch(color='blue', label='Mittelwert'))
    plt.legend(handles=plts)
    df2_months = df2_work.groupby(['Month'])['Menge'].mean().reset_index()
    #if (product_id == '722'):
    #    display(df2_months)
    
    plt.plot(df2_months['Month'], df2_months['Menge'], color='blue')
    fig.suptitle('Verkaufszahlen von Artikel %s (%s) ' % (product_id, muster))
    plt.show()

    
hortima_muster = {
    'Typ1 - Linear':[722, 173],
    'Typ2 - 1 runder Peak':[4073],
    'Typ3 - 1 spitziger Peak':[],
    'Typ4 - 2 runde Peaks':[],
    'Typ5 - 2 spitzige Peaks':[965],
    'Typ6 - Klumpenrisiko(Rest)': [6798, 982]
}
artikel_liste = []
for a, b in hortima_muster.items():
    artikel_liste = artikel_liste + b
orig_data = get_dataframe(artikel_liste)
data_month = group_by_month1(orig_data)
df2 = data_month

muster = ''

for i,v in df2.items():
    int_i = int(i)
    for mu, liste in hortima_muster.items():
        if int_i in liste:
            muster = mu
    trendbestimmung(i, v, muster)

