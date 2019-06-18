import math
import numpy as np
import matplotlib.pyplot as plt
from sanger import Sanger

X = np.genfromtxt(fname='tp2_training_dataset.csv', delimiter=',',dtype=float)
X = X[1:]
inputs = np.shape(X)[1]
n = np.shape(X)[0]

'''
for i in range(inputs):						
	mean = np.mean(X[:,i])				
	std = np.std(X[:,i])
	X[:,i] -= mean
	X[:,i] /= std
'''

red = Sanger(inputs,3)
red.train(X)
print(red.w_)