# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:21:06 2020

@author: jguerrero
"""

# Regresion lineal multiple
# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values

#dummies = pd.get_dummies(X[:, 3]).values
#X = np.concatenate([X[:, :-1], dummies[:, :-1]], axis=1)

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:, 3]  = labelencoder_x.fit_transform(X[:, 3])

# OneHotEncoder actualizado
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer([("cat", OneHotEncoder(), [3])], remainder = "passthrough")
X = onehotencoder.fit_transform(X)
X = X.astype(np.float64)

# Evitar trampa de variables Dummy (tomar todas las variables - 1)
X = X[:, 1:]

# Dividir el dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split
# random_state = 0 SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Escalado de variables
# Estandarizacion
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Escalar los datos de train y test con la misma formula
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

# Ajustar el modelo de regresion lineal multiple con los datos de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de prueba
y_pred = regression.predict(X_test)

# Construir modelo optimo utilizando la Eliminacion hacia atras
# import statsmodels.formula.api as sm
import statsmodels.api as sm
X = np.append(arr = np.ones([50, 1]).astype(int), values = X, axis = 1)
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

import statsmodels.api as sm
def backwardElimination_rsquared(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination_rsquared(X_opt, SL)

#X_opt = X[:, [0,1,2,3,4,5]].tolist()
#regressor_OLS = sm.OLS(y, x.tolist()).fit()