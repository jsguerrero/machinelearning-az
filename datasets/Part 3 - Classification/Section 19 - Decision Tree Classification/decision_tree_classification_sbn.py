# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:41:01 2020

@author: jguerrero
"""

# ARBOLES DE DECISION
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

# Ajustar la clasificacion con el conjunto de entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizacion de los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 500))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Arbol de decision (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

from sklearn.tree import plot_tree
plot_tree(classifier)