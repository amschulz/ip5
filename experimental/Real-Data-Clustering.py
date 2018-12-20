
# coding: utf-8

# In[35]:


import csv
import pandas
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def get_dataframe(value9999=False, valueOnly0=True):
    with open('Datenexporte/Datenexport_240418/Percentages2.csv', newline='') as f:
        reader = csv.reader(f, delimiter='	',) 
        attributes = next(reader)

        orig_data = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            value9999 = False
            valueOnly0 = True
            for g, column in enumerate(row):
                if(g != 6):
                    column = float(column)
                if column == '9999' or column == 9999 or column == '-9999' or column == -9999:
                    value9999 = True
                if (not (column == 0 or column == '0' or column == '')) and (not (g == 0 or g == '0')):
                    valueOnly0 = False
            if len(row) == 7 and                 not valueOnly0:
                orig_data.append(row)
        data = {}
        for j, attribute in enumerate(attributes):
            values_per_data = [x[j] for x in orig_data]
            data[attribute] = values_per_data

    data_pandas = pandas.DataFrame(data)
    return data_pandas

def remove_unnecessary_data(data_pandas):
    data = data_pandas[['R1 zu R2', 'R2 zu R3', 'R3 zu R4', 'R4 zu R5']]
    # calc_data_learn_pandas = data_pandas
    """calc_data_learn_pandas = calc_data_learn_pandas[~calc_data_learn_pandas['ï»¿ArtikelNr'].isin(['189',
     '3174',
     '4080',
     '4389',
     '5322',
     '6061',
     '7188',
     '7201',
     '7516',
     '7662',
     '7665',
     '7667',
     '7668',
     '7673',
     '7740',
     '7753'])]"""
    return data

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


# In[39]:


orig_data = get_dataframe(value9999=False, valueOnly0=True)
orig_data = remove_unnecessary_data(data_pandas=orig_data)
scaled_data = scale_data(data=orig_data, scaler=RobustScaler())
show_elbow_method(data=scaled_data, a=1, b=20)
kmeandata = run_kmeans(data=scaled_data, orig_data=orig_data, n=10)
show_data_per_label(labeled_data=kmeandata, label=2, length=5)

