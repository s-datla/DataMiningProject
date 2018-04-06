import math
import numpy as np

class LinearRegression:

    def __init__(self,lr,steps):
        self.lr = lr
        self.steps = steps

    def fit(self,input,labels,disp=False):
        X = np.insert(input,0,1,axis=1)
        input = self.normalise(input)
        self.weights = np.ones(len(X[0]))
        for i in range(0,self.steps):
            X_perm, Y_perm = self.permute(X,labels)
            for j in range(0,len(X)):
                y_hat = np.dot(X_perm[j],self.weights)
                error = Y_perm[j] - y_hat
                print error
                update = np.dot(error.T,X_perm[j])
                print update
                self.weights += self.lr * np.dot(error.T,X_perm[j])
        return self

    def permute(self,inputs,labels):
        perm_indices = np.random.permutation(len(labels))
        return inputs[perm_indices], labels[perm_indices]

    def cost(self,error,num_samples):
        return np.sum(error ** 2) / (2 * num_samples)

    def predict(self,input):
        return np.dot(np.insert(input,0,1,axis=1),self.weights)

    def clearWeights(self):
        self.weights = np.ones(9)
        return self

    def normalise(self,input):
        maxs = np.max(input,axis=0)
        return [[float(row[i])/maxs[i] for i in range(0,len(row))] for row in input]
