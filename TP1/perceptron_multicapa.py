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

    def forward(self, x):
    	mu = np.append(x,1)
    	for j in range(self.neu_):
    		self.v_[j] = self.predict(mu,self.w1_[j,])
    	for i in range(self.out_):
    		self.output_[i] = self.predict(self.v_,self.w2_[i,])
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

    def train(self, X, y, eta=0.01, epochs=100, umbral=10 ** -100):
    	self.eta = eta
    	self.epochs = epochs
    	self.errors_ = []
    	for _ in range(self.epochs):
            errors = 0
            c = list(zip(X, y))
            np.random.shuffle(c)
            for mu_i, tar in c:
            	target = np.array([tar])
            	output = self.forward(mu_i)
            	errors += self.backward(mu_i,output,target,self.eta)
            errors = errors/(np.shape(X)[0])
            if errors <= umbral:
                break
            else:
                self.errors_.append(errors)

    def sigmoid(self, X):
        return 1 / (1 + math.exp(-self.beta_*X))

    def net_input(self, X, w):
        return np.dot(X,w)

    def predict(self, X, w):
            return self.sigmoid(self.net_input(X,w))