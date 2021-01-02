#Einlesen des Datensatzes als Pandas Dataframe und Reduktion/Sortierung der Merkmale
df_to_cluster = pd.read_csv('.\SpotifyData.csv')
df_to_cluster = df_to_cluster[['id','acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']]
df_to_cluster = df_to_cluster.iloc[:,:].values

#Normalisierung der relevanten Merkmale mit dem MinMaxScaler
scaler = MinMaxScaler()
df_to_cluster[:,1:] = scaler.fit_transform(df_to_cluster[:,1:])