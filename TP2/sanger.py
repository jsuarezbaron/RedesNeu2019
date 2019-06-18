import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class Sanger(object):

	def __init__(self, DimTrain, outputs, eta=0.01):
		self.in_ = DimTrain
		self.out_ = outputs
		self.w_ = np.random.random([outputs,DimTrain])*2 - 1
		self.output_ = np.zeros(outputs)
		self.eta_ = eta
		self.delta_ = np.zeros([outputs,DimTrain])

	def forward(self, x):
		for i in range(self.out_):
			self.output_[i] = np.dot(x,self.w_[i,])

	def updateWeight(self, x):
		for j in range(self.in_):
			for i in range(self.out_):
				self.delta_[i,j] = self.eta_ * self.output_[i] * (x[j] - np.dot(self.output_[0:i+1],self.w_[0:i+1,j]))
				self.w_[i,j] += self.delta_[i,j]

	def train(self, X, epochs=100, umbral=1e-10):
		for i in range(epochs):
			x = list(X)
			np.random.shuffle(x)
			for mu in x:
				self.forward(mu)
				self.updateWeight(mu)
			if np.linalg.norm(self.delta_) < umbral:
				break