# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:35:38 2020

@author: jguerrero
"""

# LDA
# COMO IMPORTAR LAS LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTAR EL DATASET
dataset = pd.read_csv('Wine.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# DIVIDIR EL DATASET EN ENTRENAMIENTO Y PRUEBA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # random_state = 0 SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS

# ESCALADO DE VARIABLES
# ESTANDARIZACION
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# ESCALAR LOS DATOS DE TRAIN Y TEST CON LA MISMA FORMULA
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# REDUCCION DE DIMENSIONES CON LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# AJUSTAR LA REGRESION LOGISTICA CON EL CONJUNTO DE ENTRENAMIENTO
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# PREDICCION DE LA REGRESION LOGISTICA CON EL CONJUNTO DE PRUEBA
y_pred = classifier.predict(X_test)

# EVALUAR RESULTADOS CON MATRIZ DE CONFUSION
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# VISUALIZACION DE LOS RESULTADOS
colors = ('red', 'green', 'blue')

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
#X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(colors))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(colors)(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()
