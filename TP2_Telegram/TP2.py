import math
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.colors import LogNorm
from sanger import Sanger
from oja import Oja
from som import Som

data = np.genfromtxt(fname='tp2_training_dataset.csv', delimiter=',',dtype=int)
X  = np.genfromtxt(fname='tp2_training_dataset.csv', delimiter=',',dtype=float)
X = X[:,1:]
inputs = np.shape(X)[1]

"""
Escalamiento
"""
for i in range(inputs):						
	mean = np.mean(X[:,i])				
	std = np.std(X[:,i])
	X[:,i] -= mean
	if std!=0:
		X[:,i] /= std

"""
Separación de datos en entrenamiento y validación
"""
XTrain = X[0:720,]
XValid = X[720:,]
colTrain = data[0:720,0]
colValid = data[720:,0]

"""
Oja y Sanger
"""
# red = Sanger(inputs,3)
# #red = Oja(inputs,3)
# red.train(XTrain)
# w = np.transpose(red.w_)

# salidaTrain = np.dot(XTrain,w)
# salidaValid = np.dot(XValid,w)

"""
Gráficos
"""
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# col = ['k','darkgreen','green','tomato','b','darkorange','r','indigo','magenta','brown']

# for i in range(len(salidaTrain)):    
#     ax.scatter(salidaTrain[i,0], salidaTrain[i,1], salidaTrain[i,2], c = col[colTrain[i]], marker = 'o', s=50, label = 'Entrenamiento')
# for i in range(len(salidaValid)):    
#     ax.scatter(salidaValid[i,0], salidaValid[i,1], salidaValid[i,2], c = col[colValid[i]], marker = '^', s=70, label = 'Validación')

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')

# #plt.show()

# f, (p1,p2,p3) = plt.subplots(1, 3, sharex=False,sharey=False)

# for i in range(len(salidaTrain)):
#     p1.scatter(salidaTrain[i,0], salidaTrain[i,1], c = col[colTrain[i]],marker = 'o', s=50, label = "Entrenamiento")
#     p2.scatter(salidaTrain[i,0], salidaTrain[i,2], c = col[colTrain[i]],marker = 'o', s=50, label = "Entrenamiento")
#     p3.scatter(salidaTrain[i,1], salidaTrain[i,2], c = col[colTrain[i]],marker = 'o', s=50, label = "Entrenamiento")
# for i in range(len(salidaValid)):
#     p1.scatter(salidaValid[i,0], salidaValid[i,1], c = col[colValid[i]],marker = '^', s=70, label = "Validación")
#     p2.scatter(salidaValid[i,0], salidaValid[i,2], c = col[colValid[i]],marker = '^', s=70, label = "Validación")
#     p3.scatter(salidaValid[i,1], salidaValid[i,2], c = col[colValid[i]],marker = '^', s=70, label = "Validación")
    
# p1.set_xlabel('PC1')
# p1.set_ylabel('PC2')

# p2.set_xlabel('PC1')
# p2.set_ylabel('PC3')

# p3.set_xlabel('PC2')
# p3.set_ylabel('PC3')

# plt.show()

"""
SOM aplicado a todas las variables
"""

# arquitecura = (6,6)
# salida1 = np.zeros(arquitecura)
# salida2 = np.zeros(arquitecura)
# mapa = Som(inputs,arquitecura)
# mapa.train(XTrain)
# salidaColores = np.zeros([arquitecura[0],arquitecura[1],10])

# for i in range(len(XTrain)):
# 	winner = mapa.winner(XTrain[i,])
# 	salidaColores[winner[0],winner[1],colTrain[i]] += 1

# for i in range(arquitecura[0]):
# 	for j in range(arquitecura[1]):
# 		color = np.argmax(salidaColores[i,j,])
# 		salida1[i,j] = color

# salidaColores = np.zeros([arquitecura[0],arquitecura[1],10])

# for i in range(len(XValid)):
# 	winner = mapa.winner(XValid[i,])
# 	salidaColores[winner[0],winner[1],colValid[i]] += 1

# for i in range(arquitecura[0]):
# 	for j in range(arquitecura[1]):
# 		color = np.argmax(salidaColores[i,j,])
# 		salida2[i,j] = color

# #print("Entrenamiento: ",salida1)
# #print("Validación: ",salida2)

# f, (p1,p2) = plt.subplots(1, 2, sharex=False,sharey=False)

# p1.imshow(salida1, interpolation = 'nearest', vmin = 1, vmax = 9, origin = 'upper')
# p2.imshow(salida2, interpolation = 'nearest', vmin = 1, vmax = 9, origin = 'upper')

# p1.set_title('Entrenamiento')
# p2.set_title('Validación')

# plt.show()

# error = np.mean(salida1 != salida2)
# print("Error: ",error)

"""
SOM a las primeras 3 componentes principales
"""

# red3 = Sanger(inputs,3)
# red3.train(X)
# w3 = np.transpose(red3.w_)

# salida3 = np.dot(X,w3)
# salida3Train = salida3[0:720,]
# salida3Valid = salida3[720:,]

# arquitecura = (6,6)
# salida1 = np.zeros(arquitecura)
# salida2 = np.zeros(arquitecura)
# mapa3 = Som(3,arquitecura)
# mapa3.train(salida3Train)
# salidaColores = np.zeros([arquitecura[0],arquitecura[1],10])

# for i in range(len(salida3Train)):
# 	winner = mapa3.winner(salida3Train[i,])
# 	salidaColores[winner[0],winner[1],colTrain[i]] += 1

# for i in range(arquitecura[0]):
# 	for j in range(arquitecura[1]):
# 		color = np.argmax(salidaColores[i,j,])
# 		salida1[i,j] = color

# salidaColores = np.zeros([arquitecura[0],arquitecura[1],10])

# for i in range(len(salida3Valid)):
# 	winner = mapa3.winner(salida3Valid[i,])
# 	salidaColores[winner[0],winner[1],colValid[i]] += 1

# for i in range(arquitecura[0]):
# 	for j in range(arquitecura[1]):
# 		color = np.argmax(salidaColores[i,j,])
# 		salida2[i,j] = color

# # print("Entrenamiento: ",salida1)
# # print("Validación: ",salida2)

# fig = plt.figure()
# f, (p1,p2) = plt.subplots(1, 2, sharex=False,sharey=False)

# p1.imshow(salida1, interpolation = 'nearest', vmin = 1, vmax = 9, origin = 'upper')
# p2.imshow(salida2, interpolation = 'nearest', vmin = 1, vmax = 9, origin = 'upper')

# p1.set_title('Entrenamiento con 3 PCA')
# p2.set_title('Validación con 3 PCA')
# plt.show()

# error = np.mean(salida1 != salida2)
# print("Error: ",error)

"""
SOM a las primeras 9 componentes principales
"""

red9 = Sanger(inputs,9)
red9.train(X)
w9 = np.transpose(red9.w_)

salida9 = np.dot(X,w9)
salida9Train = salida9[0:720,]
salida9Valid = salida9[720:,]

arquitecura = (6,6)
salida1 = np.zeros(arquitecura)
salida2 = np.zeros(arquitecura)
mapa9 = Som(9,arquitecura)
mapa9.train(salida9Train)
salidaColores = np.zeros([arquitecura[0],arquitecura[1],10])

for i in range(len(salida9Train)):
	winner = mapa9.winner(salida9Train[i,])
	salidaColores[winner[0],winner[1],colTrain[i]] += 1

for i in range(arquitecura[0]):
	for j in range(arquitecura[1]):
		color = np.argmax(salidaColores[i,j,])
		salida1[i,j] = color

salidaColores = np.zeros([arquitecura[0],arquitecura[1],10])

for i in range(len(salida9Valid)):
	winner = mapa9.winner(salida9Valid[i,])
	salidaColores[winner[0],winner[1],colValid[i]] += 1

for i in range(arquitecura[0]):
	for j in range(arquitecura[1]):
		color = np.argmax(salidaColores[i,j,])
		salida2[i,j] = color

# print("Entrenamiento: ",salida1)
# print("Validación: ",salida2)

f, (p1,p2) = plt.subplots(1, 2, sharex=False,sharey=False)

p1.imshow(salida1, interpolation = 'nearest', vmin = 1, vmax = 9, origin = 'upper')
p2.imshow(salida2, interpolation = 'nearest', vmin = 1, vmax = 9, origin = 'upper')

p1.set_title('Entrenamiento con 9 PCA')
p2.set_title('Validación con 9 PCA')

plt.show()

error = np.mean(salida1 != salida2)
print("Error: ",error)