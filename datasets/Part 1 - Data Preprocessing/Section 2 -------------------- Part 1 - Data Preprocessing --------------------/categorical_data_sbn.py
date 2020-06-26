# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:42:06 2020

@author: jguerrero
"""

# Plantilla de datos categoricos
# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:, 0]  = labelencoder_x.fit_transform(X[:, 0])

'''from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()'''

#from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# OneHotEncoder actualizado
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
X = onehotencoder.fit_transform(X)
X = X.astype(np.float64)

# OneHotEncoder actualizado
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = "passthrough")
X = onehotencoder.fit_transform(X)
X = X.astype(np.float64)