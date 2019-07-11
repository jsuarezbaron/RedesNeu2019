import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import time

class Som(object):

	def __init__(self, DimTrain, outputs, eta0 = 0.1, sigma0 = 3, t1 = 2000, t2 = 2000):
		self.in_ = DimTrain
		self.out_ = outputs
		#self.w_ = (np.random.random([outputs[0],outputs[1],DimTrain])*2 - 1)/1000
		self.w_ = np.zeros([outputs[0],outputs[1],DimTrain])
		self.output_ = np.zeros([outputs[0],outputs[1]])
		self.eta_ = eta0
		self.sigma_ = sigma0
		self.t1_ = t1
		self.t2_ = t2

	def h(self, i, j, winner, t):
		r = np.array([i,j]) - np.array(winner)
		dist =  np.dot(r,r)
		sigma = self.sigma_ * np.exp(-t/self.t1_)
		return(np.exp(-dist/(2 * sigma**2)))

	def winner(self, x):
		dist = np.inf
		for i in range(self.out_[0]):
			for j in range(self.out_[1]):
				if(np.linalg.norm(self.w_[i,j,] - x) < dist):
					dist = np.linalg.norm(self.w_[i,j,] - x)
					winner = [i,j]
		return(winner)

	def updateWeight(self, x, t):
		for i in range(self.out_[0]):
			for j in range(self.out_[1]):
				self.w_[i,j,] = self.w_[i,j,] + self.eta_ * np.exp(-t/self.t2_) * self.h(i,j,self.winner(x),t) * (x - self.w_[i,j,])

	def train(self, X, ordIter=4000, convIter=18000, umbralOrd = 0.01, umbralConv = 1e-9):
		tamaño = np.shape(X)[0]
		print("Algoritmo Kohonen")
		time.sleep(2)
		for i in range(self.out_[0]):
			for j in range(self.out_[1]):
				self.w_[i,j,] = X[np.random.choice(tamaño),]
		for t in range(ordIter):
			print("Ordenamiento: ",t+1)
			i = np.random.choice(tamaño)
			mu = X[i,]
			self.updateWeight(mu,t)
			eta = self.eta_ * np.exp(-t/self.t2_)
			#sigma = self.sigma_ * np.exp(-t/self.t1_)
			#print(eta,sigma)
			if(eta < umbralOrd):
				break
		self.eta_ = 0.01
		self.sigma_ = 0.1
		self.t1_ = 200
		self.t2_ = 200
		for t in range(convIter):
			print("Convergencia: ",t+1)
			i = np.random.choice(tamaño)
			mu = X[i,]
			self.updateWeight(mu,t)
			eta = self.eta_ * np.exp(-t/self.t2_)
			sigma = self.sigma_ * np.exp(-t/self.t1_)
			#print(eta,sigma)
			if(eta < umbralConv or sigma < umbralConv):
				break
		print("¡Finalizado!")

