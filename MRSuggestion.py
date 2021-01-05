import numpy as np
import pandas as pd
from io import StringIO
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONProtocol
import heapq
from operator import itemgetter
import os
import ast
from datetime import datetime

current = os.getcwd()
OUTPUT_PROTOCOL = JSONProtocol

class KNNSuggestion(MRJob):
        
    def configure_args(self):
        '''
        Input args. including the address of output of MRPreprocess and the value of K.
        '''
        super(KNNSuggestion,self).configure_args()
        #model's address
        self.add_passthru_arg("--model",
                                type = str,)
        #The value of k
        self.add_passthru_arg("-k",
                                type = str,
                                default = 3)
                                     
    def load_args(self,args):
        '''
        Reads the corresponding data based on the input args.
        '''
        super(KNNSuggestion,self).load_args(args)
        #read model
        if self.options.model is None:
            #No input mod, error reported
            self.option_parser.error("please type a path")
        else:
            #read model
            self.model = {}
            with open(current+'./'+self.options.model,encoding='utf-16') as src:
                for line in src:
                    # For each line of the model file, read the corresponding labels and features and store them in the dictionary.
                    label_model, features_model = line.split('\t')
                    features_model = features_model.replace('\n', '')
                    features_model = ast.literal_eval(features_model)
                    label_model = ast.literal_eval(label_model)
                    self.model[label_model] = features_model

        #read k values
        try:
            self.k = int(self.options.k)
        except:
            self.option_parser.error("K value must be integer.")

    def __init__(self, *args, **kwargs):
        super(KNNSuggestion, self).__init__(*args, **kwargs)

    def steps(self): 
        return [MRStep(mapper=self.mapper,combiner=self.combiner,reducer=self.reducer)]

    def mapper(self,_,line):
        '''
        Mapper function. Receives each row of the test set, extracts its feature set, and calculates the K points in the training set that are closest to it.
        The class with the most K points is determined and the class corresponding to that test ssample is predicted to be that class. 
        '''
        # Extract feature set and class of test data
        data = line.split(',')
        features = [float(x) for x in data[1:-1]] #austauschen durch lambda
        features_id = data[0]
        nearest = [] #k nearest points

        #Loop through training data
        for cat in self.model:
            for point in self.model[cat]:
                point[1:] = [float(x) for x in point[1:]]
                # distance, multiplied by -1 because afterwards the heap sort will be used and needs to be ranked from largest to smallest, but the python implementation is the smallest heap, so *(-1)
                dis_euk = -1*np.linalg.norm(np.array(point[1:])-np.array(features), ord=2) #L2 Norm/eukl Distanz
                dis_man = -1*np.linalg.norm(np.array(point[1:])-np.array(features), ord=1) #L1 Norm/Manhatten Distanz
                dis_frobenius = -1*np.linalg.norm(np.array(point[1:])-np.array(features)) #Default: Frobenius Distanz
                #Make a tuple of distances, points, and categories to which they belong for easy comparison
                item = tuple([dis_euk, point[0]])
                if(dis_euk == 0):
                    continue
                if(len(nearest)<self.k):
                    # If the nearest length is less than k, append directly
                    nearest.append(item)
                    continue
                elif(len(nearest)==self.k):
                    # If the nearest length is equal to k, transform the nearest into a heap
                    heapq.heapify(nearest)
                if(dis_euk > nearest[0][0]):
                    #If the distance of the new point is less than the longest point in the nearest, the longest point is popped out and the new point enters the nearest
                    heapq.heapreplace(nearest,item)
        
        subresult = []
        for neighbour in nearest:
            nn = [neighbour[0], neighbour[1]]
            subresult.append(nn)

        yield features_id, subresult

    def combiner(self, features_id, subresult):
    
        subresults = []
        for feature in subresult:
            subresults.append(feature)
            
        yield features_id, subresults

    def reducer(self, features_id, subresults):

        for result in subresults:
            kNN = heapq.nlargest(5, *result, key=itemgetter(0))
        
        kNN = [x[1] for x in kNN]

        yield features_id, kNN

if __name__ == '__main__':
    KNNSuggestion.run()