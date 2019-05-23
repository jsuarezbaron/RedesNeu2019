import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class PerceptronMulticapa(object):

    # Constructor de la clase. 
    def __init__(self, DimTrain, neurons, outputs, beta=1):
    	self.beta_= beta
    	self.in_ = DimTrain
    	self.neu_ = neurons
    	self.out_ = outputs
    	self.w1_ = np.random.random([neurons,DimTrain+1])*2 - 1
    	self.w2_ = np.random.random([outputs,neurons+1])*2 - 1
    	self.v_ = np.zeros(neurons+1)
    	self.v_[-1] = 1
    	self.output_ = np.zeros(outputs)
    	self.delta1_ = np.zeros(neurons)
    	self.delta2_ = np.zeros(outputs)

    def forward(self, x, redEntrenada=False, w1=0, w2=0):
        if not redEntrenada:
            w1 = self.w1_
            w2 = self.w2_
        mu = np.append(x,1)
        for j in range(self.neu_):
            self.v_[j] = self.predict(mu,w1[j,])
        for i in range(self.out_):
            self.output_[i] = self.predict(self.v_,w2[i,])
        return self.output_

    def backward(self, x, output, target, eta):
    	mu = np.append(x,1)
    	for i in range(self.out_):
    		self.delta2_[i] = (self.beta_ * output[i] * (1 - output[i])) * (target[i] - output[i])
    	for i in range(self.neu_):
    		self.delta1_[i] = (self.v_[i] * (1 - self.v_[i])) * np.dot(self.delta2_,self.w2_[:,i])
    	for i in range(self.out_):
    		for j in range(self.neu_+1):
    			self.w2_[i,j] += eta * self.v_[j] * self.delta2_[i]
    	for i in range(self.neu_):
    		for j in range(self.in_+1):
    			self.w1_[i,j] += eta * mu[j] * self.delta1_[i]
    	return 0.5 * sum((target - output)**2)

    def train(self, XTrain, yTrain, XValid, yValid, eta=0.01, epochs=100, umbral=10 ** -100):
        self.eta = eta
        self.epochs = epochs
        self.errorsTrain_ = []
        self.errorsValid_ = []
        self.w1Val = 0
        self.w2Val = 0
        minVal= float("inf")
        for _ in range(self.epochs):
            errors = 0
            c = list(zip(XTrain, yTrain))
            np.random.shuffle(c)
            for mu_i, tar in c:
                if np.shape(tar)==():
                    target = np.array([tar])
                else:
                    target = np.array(tar)
                output = self.forward(mu_i)
                errors += self.backward(mu_i,output,target,self.eta)
            errors = errors/(np.shape(XTrain)[0])
            if errors <= umbral:
                break
            else:
                self.errorsTrain_.append(errors)
            errors = 0
            for x, y in zip(XValid,yValid):
                output = self.forward(x)
                if np.shape(output)==():
                    output = np.array([output])
                    y = np.array([y])
                errors += 0.5 * sum((output - y)**2)
            errors /= (np.shape(XTrain)[0])
            self.errorsValid_.append(errors)
            if self.errorsValid_[-1] < minVal:
                minVal = self.errorsValid_[-1]
                self.w1Val = self.w1_
                self.w2Val = self.w2_

    def sigmoid(self, X):
        return 1 / (1 + math.exp(-self.beta_*X))

    def net_input(self, X, w):
        return np.dot(X,w)

    def predict(self, X, w):
            return self.sigmoid(self.net_input(X,w))