
# coding: utf-8

# In[1]:


import csv
import pandas

with open('Datenexporte/Datenexport_250318/Positionen_Report.csv', newline='') as f:
    reader = csv.reader(f, delimiter='	',) 
    attributes = next(reader)
    
    orig_data = []
    artikel_count = {}
    for i in range(100000):
        orig_row = next(reader)
        if len(orig_row) == 15 and         orig_row[10] not in ['1.902599998097e+13', '1.914599998085e+13'] and         float(orig_row[10]) >= 0 and         float(orig_row[10]) < 1000000 and         float(orig_row[4]) >= 0 and         float(orig_row[4]) < 10000:
            if orig_row[0] in artikel_count.keys():
                artikel_count[orig_row[0]] += 1
            else:
                artikel_count[orig_row[0]] = 1
            orig_data.append(orig_row)
    data = {}
    for i, attribute in enumerate(attributes):
        values_per_data = [x[i] for x in orig_data]
        data[attribute] = values_per_data

# print(artikel_count)
data_pandas = pandas.DataFrame(data)
display(data_pandas)


# In[2]:


calc_data_learn_pandas = data_pandas[['Gewicht', 'Menge']][:60000]
display(calc_data_learn_pandas)
calc_data_test_pandas = data_pandas[['Gewicht', 'Menge']][60000:]
display(calc_data_test_pandas)


# In[3]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0).fit(calc_data_learn_pandas)
labels = ['Kleines Gewicht. Kleine Menge.',
          'Grosses Gewicht. Grosse Menge.',
          'Kleines Gewicht. Grosse Menge.',
          'Grosses Gewicht. kleine Menge.']
label_count = [0,0,0,0]
learn_labels = kmeans.labels_
test_labels = kmeans.predict(calc_data_test_pandas)

for i, x in enumerate(learn_labels):
    label_count[x] += 1
    #if x != 0:
    #    print(calc_data_learn_pandas['Gewicht'][i], ' ', calc_data_learn_pandas['Menge'][i])

print(label_count)
    
calc_data_learn_pandas['Kategorie'] = [labels[x] for x in learn_labels]
display(calc_data_learn_pandas)

calc_data_test_pandas['Kategorie'] = [labels[x] for x in test_labels]
display(calc_data_test_pandas)


# In[4]:


display(calc_data_test_pandas[calc_data_test_pandas.Kategorie != 'Kleines Gewicht. Kleine Menge.'])


# In[139]:


df = pandas.DataFrame(learn_labels)
import mglearn
pandas.plotting.scatter_matrix(df,cmap=mglearn.cm3)

