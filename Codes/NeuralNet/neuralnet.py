import pickle
import random as rd
import numpy as np
import math
import scipy.misc # to visualize only
import sklearn.preprocessing as sp
import pandas as pd
from sklearn.cross_validation import train_test_split
from skimage import measure
from scipy.ndimage import measurements,morphology, generate_binary_structure
from numpy import *

class NN:
    def rL(self, n):
        l = []
        for _ in range(n+1):
            l.append(rd.uniform(-1, 1)/100)
        return l

    def sig(self, val):
        return 1/(1+math.exp(-val))

    def calculateOutputs(self, X):
        self.O[0] = X

        for i in range(1, self.nLayers+2):
            #Add intercept
            out = np.concatenate((self.O[i-1],[1]))
            for j in range(len(self.O[i])):
                self.O[i][j] = self.sig(np.dot(out, self.W[i-1][j]))

    def train (self, X, Y, nLayers, mLayer, alpha, thres, maxiter):
        YO = Y
        X = np.array(X)
        Y = np.array(Y)

        Y = pd.get_dummies(Y)
        Y = Y.as_matrix()

        self.inputL = len(X[0])
        self.outL = len(Y[0])
        self.nLayers = nLayers
        self.mLayer = mLayer
        self.O = []
        self.D = []

        for _ in range(self.nLayers+2):
            self.D.append([0]*(mLayer+1))

        for _ in range(self.nLayers+2):
            self.O.append([0]*self.mLayer)
        self.O[-1] = [0]*self.outL

        # Produce random weigths
        self.W = []
        self.W.append([])
        for _ in range(self.mLayer):
            self.W[0].append(self.rL(self.inputL))
        for x in range(1, self.nLayers):
            self.W.append([])
            for y in range(self.mLayer):
                self.W[x].append(self.rL(self.mLayer))
        self.W.append([])
        for _ in range(self.outL):
            self.W[-1].append(self.rL(self.mLayer))


        lold = 0
        for _ in range(maxiter):
            dSum = [0]*self.outL
            p = np.random.permutation(len(X))
            X = X[p]
            Y = Y[p]
            for idx in range(len(X)):
                self.calculateOutputs(X[idx])

                # Compute correction for output layer
                out = self.O[-1]
                self.D[-1] =  out * np.subtract(1, out) * np.subtract(Y[idx], out)

                for i in range(self.nLayers, 0, -1):
                    for j in range(len(self.O[i])):
                        s = 0
                        for k in range(len(self.O[i+1])):
                            s += self.D[i+1][k] * self.W[i][k][j]
                        self.D[i][j] = s * self.O[i][j] * (1 - self.O[i][j])

                dSum += self.D[-1]**2
                for i in range(0, self.nLayers+1):
                    out = np.concatenate((self.O[i],[1]))
                    for j in range(len(self.O[i+1])):

                        delta = alpha * np.multiply(self.D[i+1][j], out)
                        self.W[i][j] = self.W[i][j] + delta
            print 'Iteration: ', _
            lnew = dSum.sum()
            print 'Loss: ', lnew
            if math.fabs(lnew - lold) < thres:
                print 'No significant improvment..'
                return
            lold = lnew

    def predict(self, X):
        out = []
        for x in X:
            self.calculateOutputs(x)
            out.append(np.argmax(self.O[-1]))
        return out


#-------------------------------------------------------------------------------
# Learing with raw pixel data

x = np.fromfile('train_x.bin', dtype='uint8')
x[x<255] = 0
x = x.reshape((100000,3600))
y = pd.read_csv('train_y.csv')
y = list(y['Prediction'])
subset = 500

X_train, X_test, y_train, y_test = train_test_split(x[:subset],y[:subset],test_size=0.2, random_state=3)

model = NN()
model.train(X_train,y_train,2,100,0.0001, 0.001, 300)
out = model.predict(X_test)

print (out == np.array(y_test)).sum()/float(len(y_test))


#-------------------------------------------------------------------------------
# Produce geometric features
'''
x = np.fromfile('train_x.bin', dtype='uint8')
x[x<255] = 0
x = x.reshape((100000,60,60))

feat = []
for im in x[:50000]:
    labels, nbr_objects = measure.label(im, return_num=True, neighbors=8)
    scipy.misc.imshow(labels)
    pr = measure.regionprops(labels)
    out = []
    if nbr_objects >= 2:
        for p in pr[0]:
            p = pr[0][p]
            if type(p) is int or type(p) is float:
                out.append(p)
        for p in pr[1]:
            p = pr[1][p]
            if type(p) is int or type(p) is float:
                out.append(p)
    elif nbr_objects == 1:
        for p in pr[0]:
            p = pr[0][p]
            if type(p) is int or type(p) is float:
                out.append(p)
        for p in pr[0]:
            p = pr[0][p]
            if type(p) is int or type(p) is float:
                out.append(0)
    else:
        print 'error'
        feat.append([0, 0, 0, 0,0,0])
    feat.append(out)

out = pd.DataFrame(feat)
out.to_csv('out.csv')
'''

#-------------------------------------------------------------------------------
# Cross validation with geometric features
'''
from sklearn.model_selection import KFold

feat = np.array(pd.read_csv('out.csv'))
y = pd.read_csv('train_y.csv')
y = np.array(list(y['Prediction']))

kf = KFold(n_splits=5)
res = 0
for train, test in kf.split(feat[:1000]):
    model = NN()
    model.train(feat[train], y[train],2,25,0.0001, 0.001, 300)
    out = model.predict(feat[test])
    res += (out == np.array(y[test])).sum()/float(len(y[test]))
print res/5
'''
