# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:29:53 2020

@author: jguerrero
"""

# Regresion polinomica
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

# Ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lineal_reg = LinearRegression()
lineal_reg.fit(X, y)

# Ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
#X_poly = poly_reg.fit_transform(X)
lineal_reg_2 = LinearRegression()
#lineal_reg_2.fit(X_poly, y)
lineal_reg_2.fit(poly_reg.fit_transform(X), y)

# Visualizacion de los resultados del modelo lineal
plt.scatter(X, y, color = 'red')
plt.plot(X, lineal_reg.predict(X), color = 'blue')
plt.title('Modelo de Regresion Lineal')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo en USD $')
plt.show()

# Visualizacion de los resultados del modelo polinomico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
#plt.plot(X, lineal_reg_2.predict(X_poly), color = 'blue')
plt.plot(X_grid, lineal_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Modelo de Regresion Polinomica')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo en USD $')
plt.show()

# Prediccion de nuestros modelos
lineal_reg.predict([[6.5]])
lineal_reg_2.predict(poly_reg.fit_transform([[6.5]]))