import numpy as np
import collections
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONValueProtocol
import MRKnnTrain
import heapq
import os
import json
from itertools import islice
import ast
import csv
import pandas as pd

k = 10 
model = {}
df_new = pd.DataFrame()
with open('.\model_neu.json',encoding='utf-16') as src:
    for line in src:
        label_model, features_model = line.split('\t')
        features_model = features_model.replace('\n', '')
        features_model = ast.literal_eval(features_model)
        label_model = ast.literal_eval(label_model)
        model[label_model] = features_model

#print(model.items()[0])
#print(list(model.items())[0])
#model_neu = json.loads(model)
#print(list(model_neu.items())[0])
#print(list(model.values())[0])
file = pd.read_csv('.\data.csv')
with open(".\\test_neu.csv") as src:
    for line in src:
        # Extract feature set and class
        data = line.split(',')
        label = data[-1]
        #print(data[:-1])
        print(data[-1])
        features = [float(x) for x in data[:-1]]
        #print(features)
        nearest = [] #k nearest points
        count = {} #The number corresponding to each category in nearest

        for cat in model:
            for point in model[cat]:
                # distance, multiplied by -1 because afterwards the heap sort will be used and needs to be ranked from largest to smallest, but the python implementation is the smallest heap, so *(-1)
                #print(point)
                point[1:] = [float(x) for x in point[1:]]
                #print(np.array(point[1:]))
                #print(np.array(features))
                dis = -1*np.linalg.norm(np.array(point[1:])-np.array(features)) 
                #Make a tuple of distances, points, and categories to which they belong for easy comparison
                item = tuple([dis, point[1:], cat, point[0]])
                if(len(nearest)<k):
                    # If the nearest length is less than k, append directly
                    nearest.append(item)
                    continue
                elif(len(nearest)==k):
                    # If the nearest length is equal to k, transform the nearest into a heap
                    heapq.heapify(nearest)
                if(dis > nearest[0][0]):
                    # If the distance of the new point is less than the longest point in the nearest, the longest point is popped out and the new point enters the nearest
                    heapq.heapreplace(nearest,item)
        
        #heaptemp = heapq.heappop(nearest)
        # print(range(len(nearest)))

        # for i in range(len(nearest)):
        #     temp = heapq.heappop(nearest)
        #     #print(heapq.heappop(nearest))
        #     #print(temp[2])
        #     print(count)
        #     if(temp[2] not in count):
        #         count[temp[2]] = 1
        #     else:
        #         count[temp[2]] += 1
        #     #print(count[temp[2]])
        # # of most calculated categories        
        # res = max(count, key=count.get)
        # print(res)
        # # Output true if the prediction is successful, otherwise false
        # if(res==label):
        #     predictor = 'true', 1
        # else:
        #     predictor = 'false', 1

        # print(predictor)
        
        #temp = heapq.heappop(nearest)
        # print(temp)
        for neighbour in nearest:
            #print(neighbour)
            #print(neighbour[3])
            df_new = df_new.append(file[(file['id'] == neighbour[3])])
            # df_new = file[(file['acousticness'] == neighbour[1][0]) & 
            # (file['danceability'] == neighbour[1][1]) &
            # (file['energy'] == neighbour[1][2]) &
            # (file['instrumentalness'] == neighbour[1][3]) &
            # (file['key'] == neighbour[1][4]) &
            # (file['liveness'] == neighbour[1][5]) &
            # (file['loudness'] == neighbour[1][6]) &
            # (file['mode'] == neighbour[1][7]) &
            # (file['speechiness'] == neighbour[1][8]) &
            # (file['tempo'] == neighbour[1][9]) &
            # (file['valence'] == neighbour[1][10])]

print(df_new)
#df_new.convert_dtypes()
#(df['Salary_in_1000']>=100) & (df['Age']< 60) & (df['FT_Team']
#[file]# == nearest[1][1][1].astype('float64'
#rows = [row for row in reader if row['Total_Depth'] != '0']
#rows = [row for row in file if 'acousticness'==nearest[1][1][1]]
#'acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
#       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence'
#['acousticness','danceability','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','valence']
#[956.0, 444.0, 0.19699999999999998, 435.0, 11.0, 0.0744, -17226.0, 1.0, 0.04, 80495.0, 305.0]
#        993.0, 409.0, 0.057999999999999996, 631.0, 4.0, 253.0, -19395.0, 0.0, 0.0364, 63371.0, 0.6759999999999999
#0.9940000000000001,0.379,0.0135,0.9009999999999999,8,0.0763,-28.454,1,0.0462,83.97200000000002,0.0767
print('Ihre Songvorschläge basierend auf Ihrer Eingabe: \n')

for index, song in df_new.iterrows():
    print('Name des Songs: {} \n Künstler: {} \n'.format(song['name'], song['artists']))
#print("Ihr Liedvorschlag! Songname: " + str(df_new['name']) + "  von  " + str(df_new['artists']))
#[0.9940000000000001, 379.0, 0.0135, 0.9009999999999999, 8.0, 0.0763, -28454.0, 1.0, 0.0462, 83.97200000000002, 0.0767]
#[0.9940000000000001, 0.379, 0.0135, 0.9009999999999999, 8, 0.0763, -28.454, 1, 0.0462, 83.97200000000002, 0.0767]