
# coding: utf-8

# In[112]:


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

def get_dataframe():
    starttime = datetime.datetime.now()
    df = pd.read_csv(filepath_or_buffer='Datenexporte/Datenexport_050518/Verkaufszahlen_v2.txt',
                     sep=';',
                     header=0,
                     usecols=[0,6,8,10])
    df = df.rename(index=str, columns={'fk_ArtikelBasis_ID_l':'ArtikelID'})
        
    # remove inactive products
    df = df[df['Inaktiv_b'] == 'Falsch']
    # df = df[(df['ArtikelID'] == 6) | (df['ArtikelID'] == 8)]

     # convert FaktDatum to datetime
    df['FaktDatum'] = pd.to_datetime(df['FaktDatum'], errors='coerce')
    
    # remove old datasets
    delta = date.today() - timedelta(365 * 3)
    df = df[(df['FaktDatum'] >= delta) & (df['FaktDatum'] < date(2017, 10, 1))]
    
    # group by quarter and sum up the 'Menge'
    df['Sum'] = df.groupby([df.ArtikelID, 
                            df.FaktDatum.dt.year,
                            df.FaktDatum.dt.quarter])['Menge'].transform('sum')
    df['quarter'] = df.FaktDatum.dt.quarter
    df['year'] = df.FaktDatum.dt.year
    df['quarterdate'] = df.FaktDatum - pd.tseries.offsets.QuarterEnd()
    
    max_date = df.quarterdate.max()
    min_date = df.quarterdate.min()

    # remove unnecessary columns & drop duplicate values
    df = df.drop(columns=['Inaktiv_b', 'Menge', 'FaktDatum', 'quarterdate'])
    df = df.drop_duplicates(subset=['ArtikelID', 'year', 'quarter'])
    
    # calculate amount of quaretrs between min and max date of datasets
    quarters = int(((max_date.year - min_date.year)*12 + (max_date.month - min_date.month))/3.0)
    
    # create new dataframe with all ArtikelIds
    df2 = pd.DataFrame({'ArtikelID': df['ArtikelID'].unique()})
    
    listoflabels = []
    
    
    # go through the quarter difference 
    for quart in range(0, quarters):
        q = ((min_date.quarter + quart) % 4 + 1)
        y = int((min_date.quarter + quart) / 4) + min_date.year
        s = '%sQ%s' % (y, q )
        
        # get data of df where quarter and year are the same
        data = df[(df['quarter'] == q) & (df['year'] == y)]
        
        # add new column to df2 (sum of the quarter)
        df2 = df2.merge(data[['ArtikelID', 'Sum']],
                        on=['ArtikelID'],
                        how='outer')
        df2.rename(inplace=True, columns={'Sum': s})
        listoflabels.append(s)
        
    # from the merge come NaN values. replace them with 0
    df2 = df2.fillna(0)
    
    for i, label in enumerate(listoflabels):
        if i != 0:
            labelold = listoflabels[i - 1]
            s = '%s : %s' % (labelold, label)
            df2[s] = df2[label] / df2[labelold] - 1
            conditions = [
                (df2[s] >= 1.00), # 7
                (df2[s] >= 0.80) & (df2[s] < 1.00), # 6
                (df2[s] >= 0.60) & (df2[s] < 0.80), # 5
                (df2[s] >= 0.40) & (df2[s] < 0.60), # 4
                (df2[s] >= 0.20) & (df2[s] < 0.40), # 3
                (df2[s] >= 0.10) & (df2[s] < 0.20), # 2
                (df2[s] >= 0.05) & (df2[s] < 0.10), # 1
                (df2[s] >= -0.05) & (df2[s] < 0.05), # 0 
                (df2[s] >= -0.10) & (df2[s] < -0.05), # -1
                (df2[s] >= -0.20) & (df2[s] < -0.10), # -2
                (df2[s] >= -0.40) & (df2[s] < -0.20), # -3
                (df2[s] >= -0.60) & (df2[s] < -0.40), # -4
                (df2[s] >= -0.80) & (df2[s] < -0.50), # -5
                (df2[s] >= -1.00) & (df2[s] < -0.80), # -6
                (df2[s] < -1.00) # -7
            ]
            choices = [7, 6, 5, 4, 3, 2, 1, 0,
                       -1, -2, -3, -4, -5, -6, -7]
            df2['%s Category' % (s)] = np.select(conditions, choices, default=0)
            df2 = df2.drop(columns=[labelold, s])
            if i == len(listoflabels) - 1:
                df2 = df2.drop(columns=[label])
        
    
    endtime = datetime.datetime.now()
    print('Duration of getting DataFrame: %s seconds' % (endtime-starttime).seconds)
    return df2    

def scale_data(data, scaler=StandardScaler()):
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled

def run_kmeans(data, orig_data, n=10):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
    
    label_count = []
    for i in range(n):
        label_count.append(0)

    labels = kmeans.labels_
    for i, x in enumerate(labels):
        label_count[x] += 1

    print(label_count)
    # calc_data_learn_pandas['Kategorie'] = [labels[x] for x in learn_labels]
    orig_data['Kategorie'] = [x for x in labels]
    return orig_data

def show_data_per_label(labeled_data, label=None, length=10):
    if label == None:
        display(labeled_data)
    else:
        display(labeled_data[labeled_data['Kategorie'].isin([label])][:10])
        
def show_elbow_method(data, a=1, b=25):
    kmeans = None
    ar = {}
    
    if a < 1:
        raise Exception()

    for i in range(a, b):
        kmeans  = KMeans(n_clusters=i, random_state=0).fit(data)
        ar[i] = kmeans.inertia_

    plt.figure()
    plt.plot(list(ar.keys()), list(ar.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("AR")
    plt.show()


# In[113]:


orig_data = get_dataframe()
display(orig_data)
# scaled_data = scale_data(orig_data, scaler=StandardScaler())
data=orig_data

# show_elbow_method(data=data, a=1, b=12)
# Originaldaten: Optimum zwischen 2 und 4; 
# StandardScalerDaten: Optimum zwischen 2 und 3;
# RobustScalerDaten: Optimum zwischen 3 und 4;
# alles zu unpräzise für unseren Verwendungszweck


# In[119]:


def get_relevant_columns(df):
    suffix = ' Category'
    relevant_columns = []
    columns = list(data.columns.values)
    for column in columns:
        if column.endswith(suffix, 15):
            relevant_columns.append(column)
    return relevant_columns

def fill_templates(columns):
        templ= {'hype_up': [1, 1, 1, 1, 1, 4, 5, 6],
                'hype_neutral': [0, 0, 0, 0, 0, 4, 5, 6],
                'hype_down': [-1, -1, -1, -1, -1, 4, 5, 6],
                'neutral_up': [1, 0, 1, 0, 1, 0, 0, 0],
                'neutral_neutral': [0, 0, 0, 0, 0, 0, 0, 0],
                'neutral_down': [-1, 0, -1, 0, -1, 0, 0, 0],
                'anti-hype_up': [1, 1, 1, 1, 1, -4, -5, -7],
                'anti-hype_neutral': [0, 0, 0, 0, 0, -4, -4, -7],
                'anti-hype_down': [-1, -1, -1, -1, -1, -4, -5, -6],
                'volatile_up': [6, -6, 6, -6, 7, -6, 7, -6],
                'volatile_neutral': [6, -6, 6, -6, 6, -6, 6, -6],
                'volatile_down': [-6, 6, -6, 6, -7, 7, -7, 7]}
        return templ

def select_template(row, relevant_columns, templates):
    template_keys = templates.keys()
    column_length = len(relevant_columns)
    index = 0
    best_template = None
    best_sum_euler = (column_length * 15 * 15) # 15 because of the max possible difference (currently 15) 
    for key in template_keys:
        template = templates[key]
        sum_euler_difference = 0
        for i, value in enumerate(reversed(template)):
            calc_index = column_length - i - 1
            if calc_index < 0:
                continue
            row_val = row[relevant_columns[calc_index]]
            difference = value - row_val
            euler_difference = difference * difference
            sum_euler_difference += euler_difference
        if sum_euler_difference == best_sum_euler:
            print('Template-Collision!')
        elif sum_euler_difference < best_sum_euler:
            best_sum_euler = sum_euler_difference
            best_template = key            
    return best_template
    

def set_template(df, relevant_columns, templates):
    d = df.apply(lambda row: select_template(row, relevant_columns, templates), axis=1)
    df['Template'] = d
    return df
                
relevant_columns = get_relevant_columns(data)
templates = fill_templates(relevant_columns)
data = set_template(data, relevant_columns, templates)
data


# In[130]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def linear_regression_all(data):
    data_to = data.drop(columns=['2017Q1 : 2017Q2 Category', 'Template'])
    X_train, X_test, y_train, y_test = train_test_split(data_to, data['2017Q1 : 2017Q2 Category'], random_state=0)
    lr = LinearRegression().fit(X_train, y_train)
    print('All - Score training: %s' % lr.score(X_train, y_train))
    print('All - Score test: %s' % lr.score(X_test, y_test))

def linear_regression_per_template(data, templates):
    for key in templates.keys():
        df = data.loc[data['Template'] == key]
        df_to = df.drop(columns=['2017Q1 : 2017Q2 Category', 'Template'])
        X_train, X_test, y_train, y_test = train_test_split(df_to, df['2017Q1 : 2017Q2 Category'], random_state=0)
        if len(X_train) == 0:
            continue
        lr = LinearRegression().fit(X_train, y_train)
        print('Template(%s) - Score training: %s' % (key, lr.score(X_train, y_train)))
        print('Template(%s) - Score test: %s' % (key, lr.score(X_test, y_test)))
            
linear_regression_all(data)
linear_regression_per_template(data, templates)

