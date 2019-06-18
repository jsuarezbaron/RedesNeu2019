import math
import numpy as np
import matplotlib.pyplot as plt
from perceptron_multicapa import PerceptronMulticapa

X = np.genfromtxt(fname='tp1_ej1_training.csv', delimiter=',',dtype=float, usecols=(1,2,3,4,5,6,7,8,9,10))
y = np.genfromtxt(fname='tp1_ej1_training.csv', delimiter=',',dtype=str, usecols=0)
y = np.where(y == 'M', 0.95, 0.05)		###Decidimos convertir las 'M' a 0,95 (a modo de 1) y las 'B' a 0.05 (a modo de 0) 
										###ya que la sigmoidea no alcanza los valores 1 y 0.

for i in range(10):						###Escalamos los datos de forma estandarizada
	mean = np.mean(X[:,i])				###ya que sin escalar nos daba overflow al evaluarlos en la sigmoidea
	std = np.std(X[:,i])
	X[:,i] -= mean
	X[:,i] /= std

XTrain = X[0:328,:] 					###80% training, 10% validation y 10% testing
yTrain = y[0:328]
XValid = X[328:369,:]
yValid = y[328:369]
XTest = X[369:,:]
yTest = y[369:]

np.random.seed(2)
ppn1 = PerceptronMulticapa(10,10,1)		###El primer argumento corresponde a la cantidad de entradas, 
										###el segundo corresponde a la cantidad de neuronas en la capa oculta 
										###y el tercero a la cantidad de salidas.

ppn1.train(XTrain, yTrain, XValid, yValid, epochs=1500, eta=0.1)

plt.plot(range(1, len(ppn1.errorsTrain_)+1), ppn1.errorsTrain_,color='b',label='Errores training')
plt.plot(range(1, len(ppn1.errorsValid_)+1), ppn1.errorsValid_,color='r',label='Errores validation')
plt.legend()
plt.grid()
plt.xlabel('Epocas')
plt.ylabel('Error')
plt.show()

testing = np.array([])
for i in range(len(yTest)):
	testing = np.append(testing,ppn1.forward(XTest[i,:], redEntrenada=True, w1=ppn1.w1Val, w2=ppn1.w2Val))

testing = np.where(testing > 0.5, 0.95, 0.05)

matrizConfusion = np.zeros([2,2])

for i in range(len(yTest)):
	if yTest[i] == 0.95:
		if yTest[i] == testing[i]:
			matrizConfusion[0,0] += 1			###Verdadero positivo
		else:
			matrizConfusion[0,1] += 1			###Falso negativo
	else:
		if yTest[i] == testing[i]:
			matrizConfusion[1,1] += 1			###Verdadero negativo
		else:
			matrizConfusion[1,0] += 1			###Falso positivo

print(matrizConfusion)

# X = np.genfromtxt(fname='tp1_ej2_training.csv', delimiter=',',dtype=float, usecols=(0,1,2,3,4,5,6,7))
# y = np.genfromtxt(fname='tp1_ej2_training.csv', delimiter=',',dtype=float, usecols=(8,9))

# for i in range(8):							###Escalamos los datos de manera estandarizada
# 	mean = np.mean(X[:,i])					###por el mismo motivo que el ejercicio anterior
# 	std = np.std(X[:,i])
# 	X[:,i] -= mean
# 	X[:,i] /= std

# for i in range(2):							###Decidimos usar este rescaling (min-max) 
# 	m = np.min(y[:,i])						###ya que utilizar estandarización podría 
# 	M = np.max(y[:,i])						###permitir datos por encima de 1 y por debajo de 0, 
# 	y[:,i] -= m 							###valores que la sigmoidea no alcanza.
# 	y[:,i] /= (M - m)						###Originalmente este rescaling deja los datos en [0,1],
# 	y[:,i] *= 0.90							###pero multiplicamos por 0.90 y sumamos 0.05 para que
# 	y[:,i] += 0.05							###los datos queden entre [0.05,0.95] y la sigmoidea
# 											###alcance dichos valores.

# XTrain = X[0:400,:] 						###80% training, 10% validation y 10% testing
# yTrain = y[0:400,:]
# XValid = X[400:450,:]
# yValid = y[400:450,:]
# XTest = X[450:,:]
# yTest = y[450:,:]

# np.random.seed(2)
# ppn2 = PerceptronMulticapa(8,8,2)	
# ppn2.train(XTrain, yTrain, XValid, yValid, epochs=1500, eta=1)

# plt.plot(range(1, len(ppn2.errorsTrain_)+1), ppn2.errorsTrain_,color='b',label='Errores training')
# plt.plot(range(1, len(ppn2.errorsValid_)+1), ppn2.errorsValid_,color='r',label='Errores validation')
# plt.legend()
# plt.grid()
# plt.xlabel('Epocas')
# plt.ylabel('Error')
# plt.show()

# testing = np.empty((0,2), float)
# for i in range(np.shape(yTest)[0]):
# 	testing = np.append(testing,[ppn2.forward(XTest[i,:], redEntrenada=True, w1=ppn2.w1Val, w2=ppn2.w2Val)],axis=0)

# errorTesting = np.mean(sum((yTest - testing)**2))
# print(errorTesting)