# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:10:58 2020

@author: jguerrero
"""

# K-FOLD CROSS VALIDATION
# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Social_Network_ads.csv')

X = dataset.iloc[:, [2,3]].values
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
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',
                 random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# APLICAR K-FOLD CROSS VALIDATION
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

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
