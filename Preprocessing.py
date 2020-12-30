import numpy as np
import pandas as pd
from sklearn import cluster
import os
from sklearn.preprocessing import MinMaxScaler


df_to_cluster = pd.read_csv('.\data.csv')
#this is the data to be used to initialize the centroids for the clusters of genres in the 160k+ tracks

df_to_cluster = df_to_cluster.drop(columns=['id','duration_ms','release_date','popularity','year','explicit'])
df_to_cluster = df_to_cluster[['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']] #'id', #'name','artists',

data = df_to_cluster.iloc[:,:].values

scaler = MinMaxScaler()

data[:,:] = scaler.fit_transform(data[:,:])

cluster_model = cluster.KMeans(n_clusters=10,  init='k-means++')
cluster_model.fit(data[:,:])

predict = cluster_model.predict(data[:,:])

result = np.column_stack((data, predict))

np.savetxt('clustered_data.csv', result, encoding = 'utf-8', fmt = "%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%d")#%s,