import numpy as np
import pandas as pd
from sklearn import cluster
import os
from sklearn.preprocessing import MinMaxScaler


df_to_cluster = pd.read_csv('.\data.csv')
df_to_cluster = df_to_cluster.drop(columns=['duration_ms','release_date','popularity','year','explicit'])
df_to_cluster = df_to_cluster[['id','acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']] # #'name','artists',

df_to_cluster = df_to_cluster.iloc[:,:].values

scaler = MinMaxScaler()

df_to_cluster[:,1:] = scaler.fit_transform(df_to_cluster[:,1:])


cluster_model = cluster.KMeans(n_clusters=100,  init='k-means++')
cluster_model.fit(df_to_cluster[:,1:])

predict = cluster_model.predict(df_to_cluster[:,1:])

result = np.column_stack((df_to_cluster, predict))

np.savetxt('clustered_data_100.csv', result, encoding = 'utf-8', fmt = "%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%d")