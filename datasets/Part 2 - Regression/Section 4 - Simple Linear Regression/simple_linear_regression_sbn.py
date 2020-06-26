# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:15:16 2020

@author: jguerrero
"""

# Regresion lineal simple
# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

# Dividir el dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)#, random_state = 0) SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS

# PARA REGRESION LINEAL SIMPLE NO SE REQUIERE ESCALAR DATOS
# Escalado de variables
# Estandarizacion
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Escalar los datos de train y test con la misma formula
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

# Crear modelo de regresion lineal simple con los datos de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de prueba
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo vs Años de experiencia [Conjunto de Entrenamiento]')
plt.xlabel('Años de experiencia')
plt.ylabel('Sueldo (USD $)')
plt.show()

# Visualizar los resultados de prueba
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
#plt.plot(X_test, y_pred, color = 'blue')
plt.title('Sueldo vs Años de experiencia [Conjunto de Test]')
plt.xlabel('Años de experiencia')
plt.ylabel('Sueldo (USD $)')
plt.show()