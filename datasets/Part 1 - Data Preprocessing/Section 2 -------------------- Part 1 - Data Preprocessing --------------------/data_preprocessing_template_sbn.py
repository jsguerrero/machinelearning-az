# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:16:57 2020

@author: jguerrero
"""

# Plantilla de pre procesado
# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Dividir el dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)#, random_state = 0) SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS

# Escalado de variables
# Estandarizacion
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Escalar los datos de train y test con la misma formula
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

