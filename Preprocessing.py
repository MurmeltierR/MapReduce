import numpy as np
import pandas as pd
from sklearn import cluster
import os
from sklearn.preprocessing import MinMaxScaler


df_to_cluster = pd.read_csv('.\data.csv')
df_to_cluster = df_to_cluster.drop(columns=['duration_ms','release_date','popularity','year','explicit'])
df_to_cluster = df_to_cluster[['id','acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']] # #'name','artists',

data = df_to_cluster.iloc[:,2:].values

scaler = MinMaxScaler()

data[:,2:] = scaler.fit_transform(data[:,2:])

cluster_model = cluster.KMeans(n_clusters=45,  init='k-means++')
cluster_model.fit(data[:,2:])

predict = cluster_model.predict(data[:,2:])

result = np.column_stack((data, predict))

np.savetxt('clustered_data_45.csv', result, encoding = 'utf-8', fmt = "%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%d")#%s,