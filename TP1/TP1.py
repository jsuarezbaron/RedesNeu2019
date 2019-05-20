import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

from perceptron_multicapa import PerceptronMulticapa
from numpy import genfromtxt

X = genfromtxt(fname='tp1_ej1_training.csv', delimiter=',',dtype=float, usecols=(1,2,3,4,5,6,7,8,9,10))
y = genfromtxt(fname='tp1_ej1_training.csv', delimiter=',',dtype=str, usecols=0)
y = np.where(y == 'M', 1, 0)

for i in range(10):
	mean = np.mean(X[:,i])
	std = np.std(X[:,i])
	X[:,i] -= mean
	X[:,i] /= std

ppn = PerceptronMulticapa(10,10,1)
ppn.train(X, y, epochs=500, eta=0.1)

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Clasificaciones erroneas')
plt.show()
