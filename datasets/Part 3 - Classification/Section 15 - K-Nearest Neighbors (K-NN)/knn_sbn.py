# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:12:38 2020

@author: jguerrero
"""

# K vecinos mas cercanos KNN
# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Social_Network_ads.csv')

X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# Dividir el dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) # random_state = 0 SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS

# Escalado de variables
# Estandarizacion
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Escalar los datos de train y test con la misma formula
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar la clasificacion con el conjunto de entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,
                                  metric = 'minkowski',
                                  p = 2)
classifier.fit(X_train, y_train)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizacion de los resultados
from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_train
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
#X_set, y_set = X_test, y_test
X1_n, X2_n = np.meshgrid(np.arange(start = X_set[:, 0].min(),
                               stop = X_set[:, 0].max() + 1,
                               step = (abs(X_set[:, 0].min()) + abs(X_set[:, 0].max() + 1)) / 500),
                               #step = 1),
                     np.arange(start = X_set[:, 1].min(),
                               stop = X_set[:, 1].max() + 1,
                               step = (abs(X_set[:, 1].min()) + abs(X_set[:, 1].max() + 1)) / 500))
                               #step = 10000))
#X_set, y_set = sc_X.inverse_transform(X_train), y_train
X_set, y_set = sc_X.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min(),
                               stop = X_set[:, 0].max() + 10,
                               step = (abs(X_set[:, 0].max() + 10 - abs(X_set[:, 0].min())) / 500)),
                     np.arange(start = X_set[:, 1].min(),
                               stop = X_set[:, 1].max() + 10000,
                               #step = 0.01))
                               step = (abs(X_set[:, 1].max() + 10000 - abs(X_set[:, 1].min())) / 500)))
plt.contourf(X1,
             X2,
             classifier.predict(np.array([X1_n.ravel(), X2_n.ravel()]).T).reshape(X1_n.shape),
             alpha = 0.75,
             cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],
                X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
