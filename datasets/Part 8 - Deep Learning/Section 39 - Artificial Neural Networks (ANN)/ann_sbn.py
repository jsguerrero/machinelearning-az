# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:55:24 2020

@author: jguerrero
"""

# REDES NEURONALES ARTIFICIALES
# THEANO
# pip3 install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# INSTALACION DE TENSORFLOW Y KERAS
# conda install -c conda-forge keras

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

# ESCALADO DE VARIABLES
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# DIVIDIR EL DATASET EN ENTRENAMIENTO Y PRUEBA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # random_state = 0 SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS

# PARTE 2 - ALGORITMO DE RED NEURONAL ARTIFICIAL
from keras.models import Sequential
from keras.layers import Dense

# INICIALIZACION DE RNA
classifier = Sequential()

# AGREGAR CAPA DE ENTRADA Y PRIMERA CAPA OCULTA
# LA CAPA DE ENTRADA ES LA CANTIDAD DE CARACTERISTICAS
# input_dim = X.shape[1] = 11
# PARA LA CANTIDAD DE NEURONAS EN LA CAPA OCULTA
# SE PUEDE DEFINIR CON LA MEDIA ENTRE ENTRADAS Y SALIDAS
# units = 11 + 1 / 2 = 6
# LOS PESOS INCIALES SE DEFINEN CON UNA DISTRIBUCION UNIFORME
# kernel_initializer = 'uniform'
# LA FUNCION DE ACTIVACION PARA ESTA CAPA SERA RECTIFICADOR
# activation = 'relu'
# LA MAYORIA DE ESTOS PARAMETROS SON A CRITERIO PERSONAL/PRUEBAS
classifier.add(Dense(input_dim = X.shape[1], units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# AGREGAR SEGUNDA CAPA OCULTA
# SABE QUE LAS ENTRADAS SON 6, QUE REPRESENTAN LA SALIDA DE LA CAPA ANTERIOR
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# AGREGAR LA CAPA DE SALIDA
# DEBIDO A QUE SE BUSCA ENCONTRAR LA PROBABILIDAD DE UN SUCESO
# SE OPTA POR UNA ACTIVACION SIGMOIDE
# activation = 'sigmoid'
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# COMPILAR LA RED NEURONAL ARTIFICIAL
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# AJUSTE DE RED NEURONAL ARTIFICIAL A LOS DATOS DE ENTRENAMIENTO
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

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