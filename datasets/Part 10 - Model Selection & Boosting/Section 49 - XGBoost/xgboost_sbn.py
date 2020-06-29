# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:45:47 2020

@author: jguerrero
"""

# XGBOOST
# https://xgboost.readthedocs.io/en/latest/build.html
# https://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform/39811079#39811079


# PARTE 1 - PRE PROCESADO DE DATOS
# LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DATASET
dataset = pd.read_csv('Churn_Modelling.csv')

# VARIABLES DEPENDIENTES
dataset_X = dataset.iloc[:, 3:-1]
# VARIABLE INDEPENDIENTE
target = dataset.iloc[:, -1]
# COLUMNAS NUMERICAS
num_columns = list(dataset_X._get_numeric_data().columns)
# COLUMNAS CATEGORICAS
cat_columns = list(set(dataset_X.columns) - set(num_columns))

# FUNCION PARA CODIFICAR COLUMNAS DE UNA DATASET
# SE ELIMINA LA PRIMER COLUMNA RESULTANTE DE CADA CODIFICACION
# PARA EVITAR LA MULTICOLINEALIDAD
def categorical_encode(dataset, columns):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import make_column_transformer
    # ONEHOTENCODER ACTUALIZADO
    encoder = OneHotEncoder(sparse=False, drop = 'first')
    preprocess = make_column_transformer((encoder, columns))
    transform = pd.DataFrame(preprocess.fit_transform(dataset[columns]), columns = preprocess.get_feature_names())
    return transform

# SE TRANSFORMAN LAS VARIABLES CATEGORICAS DEL dataset_X
# SI FUERA NECESARIO SE PUEDE REALIZAR LO MISMO CON target
categorical_transform = categorical_encode(dataset_X, cat_columns)

# SE JUNTA DE NUEVO EL DATASET
# EN EL ORDEN DE COLUMNAS NUMERICAS, CATEGORICAS, OBJETIVO
dataset_transform = dataset_X[num_columns].join(
   pd.DataFrame(data = categorical_transform, index = categorical_transform.index, columns = categorical_transform.columns)
)
dataset_transform = dataset_transform.join(target)

# MATRIZ DE CARACTERISTICAS
X = dataset_transform.iloc[:, :-1].values
# VARIABLE A PREDECIR
y = dataset_transform.iloc[:, -1].values

# DIVIDIR EL DATASET EN ENTRENAMIENTO Y PRUEBA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # random_state = 0 SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS

# Ajustar el modelo XGBoost al Conjunto de Entrenamiento
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# PARTE 3 - CALCULAR PREDICCIONES FINALES Y EVALUAR EL MODELO
# PREDICCION DE LA CLASIFICACION CON EL CONJUNTO DE PRUEBA
y_pred = classifier.predict(X_test)

# CONVERTIR DE LA PROPABILIDAD A UNA RESPUESTA
y_pred = (y_pred > 0.5)

# EVALUAR RESULTADOS CON MATRIZ DE CONFUSION
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# APLICAR K-FOLD CROSS VALIDATION
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()