# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:41:02 2020

@author: jguerrero
"""

# REDES NEURONALES CONVOLUCIONALES
# THEANO
# pip3 install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# INSTALACION DE TENSORFLOW Y KERAS
# conda install -c conda-forge keras

# PARTE 1 - ALGORITMO DE RED NEURONAL CONVOLUCIONAL
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# INICIALIZACION DE RNC
classifier = Sequential()

# AGREGAR CAPA DE CONVOLUCION
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# AGREGAR CAPA DE MAX POLLING
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# AGREGAR SEGUNDA CAPA DE CONVOLUCION
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))

# AGREGAR SEGUNDA CAPA DE MAX POLLING
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# AGREGAR CAPA DE FLATTENING
classifier.add(Flatten())

# AGREGAR PRIMERA CAPA OCULTA
# LA CANTIDAD DE NEURONAS EN LA CAPA OCULTA
# SE PUEDE DEFINIR DE FORMA ARBITRARIA
# units = 128
# LA FUNCION DE ACTIVACION PARA ESTA CAPA SERA RECTIFICADOR
# activation = 'relu'
# LA MAYORIA DE ESTOS PARAMETROS SON A CRITERIO PERSONAL/PRUEBAS
classifier.add(Dense(units = 128, activation = 'relu'))

# AGREGAR CAPA DE SALIDA
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# COMPILAR LA RED NEURONAL CONVOLUCIONAL
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# PARTE 2 AJUSTAR LA RNC A LAS IMAGENES DE ENTRENAMIENTO
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(train_dataset,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_dataset,
                         validation_steps=2000,
                         use_multiprocessing = True,
                         workers=-1)


# PARTE 3 - CALCULAR PREDICCIONES FINALES Y EVALUAR EL MODELO
import numpy as np
import cv2
img = cv2.imread('dataset/test_image/cat.4010.jpg') # leemos la imagen
img = cv2.resize(img,(64,64)) # hacemos un resize de la imagen para que coincida con la del modelo
img = np.reshape(img,[1,64,64,3])
classes = model.predict_classes(img)





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