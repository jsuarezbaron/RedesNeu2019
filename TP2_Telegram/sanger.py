import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(20)

class Sanger(object):

	def __init__(self, DimTrain, outputs, eta=0.001):
		self.in_ = DimTrain
		self.out_ = outputs
		self.w_ = (np.random.random([outputs,DimTrain])*2 - 1)/10000
		self.output_ = np.zeros(outputs)
		self.eta_ = eta
		self.delta_ = np.zeros([outputs,DimTrain])

	def forward(self, x):
		for i in range(self.out_):
			self.output_[i] = np.dot(x,self.w_[i,])
		#self.output_ = np.dot(x,self.w_)

	def updateWeight(self, x):
		for j in range(self.in_):
			for i in range(self.out_):
				self.delta_[i,j] = self.eta_ * self.output_[i] * (x[j] - np.dot(self.output_[0:i+1],self.w_[0:i+1,j]))
				self.w_[i,j] += self.delta_[i,j]

	def train(self, X, epochs=5, umbral=1e-5):
		tamaño = np.shape(X)[0]
		print("Regla Sanger")
		for t in range(epochs):
			print("Epoca: ",t+1)
			self.eta_ -= t/10000
			for i in range(tamaño):
				mu = X[i,]
				self.forward(mu)
				self.updateWeight(mu)
				#print(np.linalg.norm(self.delta_))
				if(np.linalg.norm(self.delta_)<umbral):
					break
				#print(np.linalg.norm(self.delta_))
		print("¡Finalizado!")
