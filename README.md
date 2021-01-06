# Spotify Songempfehlung

# Ausgangspunkt sind die im Format der Spotify API vorliegenden Daten --> SpotifyData.csv

# 1. Schritt: Die Spotify Daten werden mithilfe der Preprocessing.py vorverarbeitet. Dies umfasst die Skalierung der Daten und ein durch ein kMeans-Verfahren erzieltes labeln der Daten.
# Die Preprocessing.py speichert die Daten dann als clustered_data_{Anzahl_Cluster}.csv ab

# Für die spätere Verarbeitung sowie die Evaluation werden die Daten der clustered_data_{Anzahl_Cluster}.csv mittels MRKnnTrain.py in ein Key-Value-Format transformiert.
# Aufruf bspw.: python .\MRKnnTrain.py clustered_data_100.csv > model_100.json
# Die einzelnen Datenpunkte werden anhand ihres Labels sortiert.
# Format (K,V): Label, [Datenpunkte/ID]

# Das MRSuggestion.py Skript nimmt jede Zeile des Testdatensatzes (repräsentiert jeweils einen Song) entgegen (bspw. test_100.csv) und gibt jeweils eine festgelegte Anzahl an Songempfehlungen aus (kNN). 
# Aufruf bspw.: python .\MRSuggestion.py --model model_100.json -k 10 test_100.csv > output.json 
# Der Output kann als output.json abgespeichert werden.
# Format (K,V): ID, [IDs] 