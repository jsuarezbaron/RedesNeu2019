import numpy as np
import matplotlib.pyplot as plt
import math

class PerceptronMulticapa(object):

    # Constructor de la clase. 
    def __init__(self, DimTrain, neurons, outputs):
    	self.in_ = DimTrain
    	self.neu_ = neurons
    	self.out_ = output
    	self.w1_ = np.random.random([neurons,DimTrain+1])
    	self.w2_ = np.random.random([outputs,neurons+1])
    	self.v_ = np.zeros(neurons+1)
    	self.v_[-1] = 1
    	self.output_ = np.zeros(output)
    	self.delta1_ = np.zeros(neurons)
    	self.delta2_ = np.zeros(outputs)

    def forward(self, mu,beta):
    	np.append(mu,1)
        for j in range(self.neu_-1):
            self.v_[j] = self.predict(mu,self.w1_[j,],beta)
        for i in range(self.out_):
            self.output_[i] = self.predict(self.v_,self.w2_[i,])
        return self.output_

    def backward(self, mu, output, target, eta, beta):
    	for i in range(self.out_):
    		for j in range(self.neu_):
    			self.delta2_[i] = (2 * output[i] * (1 - output[i])) * (target[i] - output[i])
    			self.w2_[i,j] += eta * self.delta2_[i] * self.v_[j]
    	for i in range(self.neu_-1):
    		for j in range(self.in_):
    			self.delta1_[i] = (2 * self.v_[i] *(1 - self.v_[i])) * np.dot(self.delta2_,self.w2_[,i])
    			self.w1_[i,j] += eta * self.delta1_[i] * mu[j]
    	return 0.5 * sum((target - output)**2)
    
	def train(self, X, y, eta=0.01, epochs=100, umbral=10 ** -10):
        self.eta = eta
        self.epochs = epochs
      	self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            c = list(zip(X, y))
            np.random.shuffle(c)
            for mu_i, target in c:
            	target = np.array(target)
            	output = forward(mu_i)
            	errors += backward(mu_i,output,target, self.eta)
            if errors <= umbral:
                break
            else:
                self.errors_.append(errors)

    def sigmoid(self, X, beta):
        return 1 / (1 + math.exp(- beta*X))

    def net_input(self, X, w):
        return np.dot(X,w)

    def predict(self, X, w):
            return self.sigmoid(self.net_input(X,w))
    
    