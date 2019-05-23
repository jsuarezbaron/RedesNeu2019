import math
import numpy as np
import matplotlib.pyplot as plt
from perceptron_multicapa import PerceptronMulticapa

X = np.genfromtxt(fname='tp1_ej1_training.csv', delimiter=',',dtype=float, usecols=(1,2,3,4,5,6,7,8,9,10))
y = np.genfromtxt(fname='tp1_ej1_training.csv', delimiter=',',dtype=str, usecols=0)
y = np.where(y == 'M', 0.95, 0.05)	###Decidimos convertir las 'M' a 0,95 (a modo de 1) y las 'B' a 0.05 (a mode de 0) ya que la sigmoidea no alcanza los valores 1 y 0.

for i in range(10):
	mean = np.mean(X[:,i])
	std = np.std(X[:,i])
	X[:,i] -= mean
	X[:,i] /= std

XTrain = X[0:328,:] ###80% training, 10% validation y 10% testing
yTrain = y[0:328]
XValid = X[328:369,:]
yValid = y[328:369]
XTest = X[369:,:]
yTest = y[369:]

np.random.seed(2)
ppn1 = PerceptronMulticapa(10,10,1)		###El primer argumento corresponde a la cantidad de entradas, 
###el segundo corresponde a la cantidad de neuronas de la capa oculta y el tercero a la cantidad de salidas.
ppn1.train(XTrain, yTrain, XValid, yValid, epochs=500, eta=0.1, umbral=10**-10)

plt.plot(range(1, len(ppn1.errorsTrain_)+1), ppn1.errorsTrain_,color='b',label='Errores training')
plt.plot(range(1, len(ppn1.errorsValid_)+1), ppn1.errorsValid_,color='r',label='Errores validation')
plt.legend()
plt.grid()
plt.xlabel('Epocas')
plt.ylabel('Clasificaciones erroneas')
plt.show()

testing = np.array([])
for i in range(len(yTest)):
	testing = np.append(testing,ppn1.forward(XTest[i,:], redEntrenada=True, w1=ppn1.w1Val, w2=ppn1.w2Val))

testing = np.where(testing > 0.5, 0.95, 0.05)

matrizConfusion = np.zeros([2,2])

for i in range(len(yTest)):
	if yTest[i] == 0.95:
		if yTest[i] == testing[i]:
			matrizConfusion[0,0] += 1
		else:
			matrizConfusion[0,1] += 1
	else:
		if yTest[i] == testing[i]:
			matrizConfusion[1,1] += 1
		else:
			matrizConfusion[1,0] += 1

print(matrizConfusion)

# X = np.genfromtxt(fname='tp1_ej2_training.csv', delimiter=',',dtype=float, usecols=(0,1,2,3,4,5,6,7))
# y = np.genfromtxt(fname='tp1_ej2_training.csv', delimiter=',',dtype=float, usecols=(8,9))

# for i in range(8):
# 	mean = np.mean(X[:,i])
# 	std = np.std(X[:,i])
# 	X[:,i] -= mean
# 	X[:,i] /= std

# for i in range(2):	###Decidimos usar este rescaling (min-max) ya que utilizar estandarización podría permitir datos por encima de 1 y por debajo de 0, valores que la sigmoidea no alcanza
# 	m = np.min(y[:,i])
# 	M = np.max(y[:,i])
# 	y[:,i] -= m
# 	y[:,i] /= (M - m)

# plt.plot()
# ppn2 = PerceptronMulticapa(8,8,2)
# ppn2.train(X, y, epochs=1000, eta=0.01, umbral=0.007)

# plt.plot(range(1, len(ppn2.errors_)+1), ppn2.errors_, marker='o')
# plt.xlabel('Epocas')
# plt.ylabel('Clasificaciones erroneas')
# plt.show()