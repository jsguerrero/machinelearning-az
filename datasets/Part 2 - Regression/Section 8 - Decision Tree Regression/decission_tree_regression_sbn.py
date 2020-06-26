# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:53:31 2020

@author: jguerrero
"""

# Regresion con arboles de decision
# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Dividir el dataset en entrenamiento y prueba
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)#, random_state = 0) SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS'''

# Escalado de variables
# Estandarizacion
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Escalar los datos de train y test con la misma formula
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

# Ajustar el arbol de regresion al dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, y)

# Prediccion del arbol de regresion
y_pred = regression.predict([[6.5]])

# Visualizacion de los resultados
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)
X_grid = X
plt.scatter(X, y, color = 'red')
#plt.plot(X, lineal_reg_2.predict(X_poly), color = 'blue')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title('Modelo de Regresion')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo en USD $')
plt.show()