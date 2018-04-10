import math
import numpy as np

class LinearRegression:

    def __init__(self,lr,steps):
        self.lr = lr
        self.steps = steps

    def fit(self,input,labels):
        self.weights = []
        input = self.normalise(input)
        X = np.insert(input,0,1,axis=1)
        for label in np.unique(labels):
            curr_weights = np.ones(len(X[0]))
            modified_y = [1 if y == label else 0 for y in labels]
            for i in range(0,self.steps):
                y_hat = np.dot(X,curr_weights)
                error = modified_y - y_hat
                # print error
                curr_weights += self.lr * np.dot(error.T,X)
            self.weights += [curr_weights]
        return self

    # def permute(self,inputs,labels):
    #     perm_indices = np.random.permutation(len(labels))
    #     return inputs[perm_indices], labels[perm_indices]

    def cost(self,error,num_samples):
        return np.sum(error ** 2) / (2 * num_samples)

    # def predict(self,input):
    #     return np.dot(np.insert(input,0,1,axis=1),self.weights)

    def clearWeights(self):
        self.weights = np.ones(9)
        return self

    def predict(self,input):
        input = self.normalise(input)
        X = np.insert(input,0,1,axis=1)
        print "Predicting {} classes".format(len(self.weights))
        predicted = []
        print len(X)
        for i in range(0,len(X)):
            predicted += [self.cost(X[i])]
        return predicted

    def cost(self,input):
        return min([((i-np.dot(input,self.weights[i])) ** 2,i) for i in range(0,4)])[1]

    def normalise(self,input):
        maxs = np.max(input,axis=0)
        return [[float(row[i])/maxs[i] for i in range(0,len(row))] for row in input]
