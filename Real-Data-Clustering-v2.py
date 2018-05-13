
# coding: utf-8

# In[ ]:


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

def remove_unnecessary_data(data_pandas=None):
    #calc_data_learn_pandas = data_pandas[['R1 zu R2', 'R2 zu R3', 'R3 zu R4']]
    calc_data_learn_pandas = data_pandas
    calc_data_learn_pandas = calc_data_learn_pandas[~calc_data_learn_pandas['ï»¿ArtikelNr'].isin(['189',
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
     '7753'])]
    return calc_data_learn_pandas

def scale_data(calc_data_learn_pandas, scaler=StandardScaler()):
    scaler.fit(calc_data_learn_pandas)
    learn_scaled = scaler.transform(calc_data_learn_pandas)
    return learn_scaled

def run_kmeans(learn_scaled, n=10):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(learn_scaled)
    
    label_count = []
    for i in range(10):
        label_count.append(0)

    learn_labels = kmeans.labels_
    for i, x in enumerate(learn_labels):
        label_count[x] += 1

    print(label_count)
    # calc_data_learn_pandas['Kategorie'] = [labels[x] for x in learn_labels]
    calc_data_learn_pandas['Kategorie'] = [x for x in learn_labels]
    return calc_data_learn_pandas

def show_data_per_label(calc_data_learn_pandas, label=None):
    if label == None:
        display(calc_data_learn_pandas)
    else:
        display(calc_data_learn_pandas[calc_data_learn_pandas['Kategorie'].isin([label])])
        
def show_elbow_method(learn_scaled):
    kmeans = None
    ar = {}

    for i in range(1, 25):
        kmeans  = KMeans(n_clusters=i, random_state=0).fit(learn_scaled)
        ar[i] = kmeans.inertia_

    plt.figure()
    plt.plot(list(ar.keys()), list(ar.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("AR")
    plt.show()
    
data = get_dataframe()
# data = remove_unnecessary_data(data)
scale_data(calc_data_learn_pandas=data)
kmeandata = run_kmeans(data)
# show_data_per_label(kmeandata, 2)
# show_elbow_method(data)

