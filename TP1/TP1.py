import math
import numpy as np
import matplotlib.pyplot as plt
from perceptron_multicapa import PerceptronMulticapa

# X1 = np.genfromtxt(fname='tp1_ej1_training.csv', delimiter=',',dtype=float, usecols=(1,2,3,4,5,6,7,8,9,10))
# y1 = np.genfromtxt(fname='tp1_ej1_training.csv', delimiter=',',dtype=str, usecols=0)
# y1 = np.where(y1 == 'M', 0.95, 0.05)

# for i in range(10):
# 	mean = np.mean(X1[:,i])
# 	std = np.std(X1[:,i])
# 	X1[:,i] -= mean
# 	X1[:,i] /= std

# ppn1 = PerceptronMulticapa(10,10,1)
# ppn1.train(X1, y1, epochs=500, eta=0.1, umbral=0.005)

# plt.plot(range(1, len(ppn1.errors_)+1), ppn1.errors_, marker='o')
# plt.xlabel('Epocas')
# plt.ylabel('Clasificaciones erroneas')
# plt.show()

X2 = np.genfromtxt(fname='tp1_ej2_training.csv', delimiter=',',dtype=float, usecols=(0,1,2,3,4,5,6,7))    )
y2 = np.genfromtxt(fname='tp1_ej2_training.csv', delimiter=',',dtype=float, usecols=(8,9))

for i in range(8):
	mean = np.mean(X2[:,i])
	std = np.std(X2[:,i])
	X2[:,i] -= mean
	X2[:,i] /= std

for i in range(2):
	m = np.min(y2[:,i])
	M = np.max(y2[:,i])
	y2[:,i] -= m
	y2[:,i] /= (M - m)

plt.plot()
ppn2 = PerceptronMulticapa(8,8,2,beta=0.5)
ppn2.train(X2, y2, epochs=800, eta=0.01, umbral=0.008)

plt.plot(range(1, len(ppn2.errors_)+1), ppn2.errors_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Clasificaciones erroneas')
plt.show()
