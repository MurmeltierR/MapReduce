import numpy as np
import pandas as pd
from sklearn import cluster
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df_to_cluster = pd.read_csv('.\SpotifyGenres\data\data.csv')

df_to_cluster = df_to_cluster.drop(columns=['duration_ms','release_date','popularity','year','id','explicit'])
df_to_cluster = df_to_cluster[['name','artists','acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']]

data = df_to_cluster.iloc[:,:].values

scaler = MinMaxScaler() # You use MinMaxScaler when you do not assume that the shape of all your features follows a normal distribution otherwise StandardScaler

data[:,2:] = scaler.fit_transform(data[:,2:])

sse = []
silhouette_coefficients = []
for k in range(2, 30):
   cluster_model = cluster.KMeans(n_clusters=k,  init='k-means++')
   cluster_model.fit(data[:,2:])
   score = silhouette_score(data[:,2:], cluster_model.labels_)
   sse.append(cluster_model.inertia_)
   silhouette_coefficients.append(score)
   print(k)


plt.style.use("fivethirtyeight")
plt.plot(range(2, 30), silhouette_coefficients)
plt.xticks(range(2, 30))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

plt.style.use("fivethirtyeight")
plt.plot(range(2, 30), sse)
plt.xticks(range(2, 30))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# cluster_model = cluster.KMeans(n_clusters=k,  init='k-means++')
# cluster_model.fit(data[:,2:])

# predict=cluster_model.predict(data[:,2:])
# data['NEW_COLUMN'] = pd.Series(predict, index=df.index)