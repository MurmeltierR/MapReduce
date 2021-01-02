from sklearn.metrics import silhouette_score
from sklearn import cluster

sse = []
silhouette_coefficients = []
for k in range(2, 30):
   cluster_model = cluster.KMeans(n_clusters=k,  init='k-means++')
   cluster_model.fit(data[:,2:])
   score = silhouette_score(data[:,2:], cluster_model.labels_)
   sse.append(cluster_model.inertia_)
   silhouette_coefficients.append(score)