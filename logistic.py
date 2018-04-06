import math
import numpy as np

class LogisticRegression:

    def __init__(self,lr=0.00095,steps=100):
        self.lr = lr;
        self.steps = steps

    def sigmoid(self,values):
        values = np.array(values,dtype=np.float128)
        return 1.0/(1 + np.exp(-values))

    def logLikelihood(self,input,weights,labels):
        y_hat = np.dat(input,weights)
        return np.sum(labels * y_hat - math.log(1 + np.exp(y_hat)))

    def normalise(self,input):
        maxs = np.max(input,axis=0)
        return [[float(row[i])/maxs[i] for i in range(0,len(row))] for row in input]

    def fit(self,input,labels,disp=False):
        self.weights = []
        input = self.normalise(input)
        X = np.insert(input,0,1,axis=1)
        for label in np.unique(labels):
            curr_weights = np.ones(len(X[0]))
            modified_y = [1 if y == label else 0 for y in labels]
            for i in range(0,self.steps):
                y_hat = np.dot(X,curr_weights)
                prob_predict = self.sigmoid(y_hat)
                error = modified_y - prob_predict
                curr_weights += self.lr * np.dot(error.T,X)
                if (disp == True):
                    if (i % 10000 == 0):
                        print logLikelihood(X,curr_weights,labels)
            self.weights += [curr_weights]
        return self

    def predict(self,input):
        input = self.normalise(input)
        X = np.insert(input,0,1,axis=1)
        print "Predicting {} classes".format(len(self.weights))
        predicted = []
        print len(X)
        for i in range(0,len(X)):
            temp = []
            for j in range(0,len(self.weights)):
                temp += [(np.dot(X[i],self.weights[j]),j)]
            predicted += [max(temp)[1]]
        return predicted

    def clearWeights(self):
        self.weights = []
