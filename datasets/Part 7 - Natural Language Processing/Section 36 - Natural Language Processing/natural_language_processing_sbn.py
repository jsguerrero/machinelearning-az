# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:24:41 2020

@author: jguerrero
"""

# Natural Language Processing

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Limpieza de texto
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(0, len(dataset)):
    review = re.sub("[^a-zA-z]", " ", dataset["Review"][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)
    
# Crear bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Dividir el dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # random_state = 0 SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS

# Algoritmo de clasificacion Logistic Regression
# Ajustar la regresion logistica con el conjunto de entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de la regresion logistica con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[1, 0]
FN = conf_matrix[1, 1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
print("----------------------------------")
print("Logistic Regression")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)

# Algoritmo de clasificacion SVM
# Ajustar la clasificacion con el conjunto de entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',
                 random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[1, 0]
FN = conf_matrix[1, 1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
print("----------------------------------")
print("SVM")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)

# Algoritmo de clasificacion Random Forest
# Ajustar la clasificacion con el conjunto de entrenamiento
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
classifier.fit(X_train, y_train)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[1, 0]
FN = conf_matrix[1, 1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
print("----------------------------------")
print("RandomForest")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)

# Algoritmo de clasificacion KNN
# Ajustar la clasificacion con el conjunto de entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,
                                  metric = 'minkowski',
                                  p = 2)
classifier.fit(X_train, y_train)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[1, 0]
FN = conf_matrix[1, 1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
print("----------------------------------")
print("KNN")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)

# Algoritmo de clasificacion Kernel SVM
# Ajustar la clasificacion con el conjunto de entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',
                 random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[1, 0]
FN = conf_matrix[1, 1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
print("----------------------------------")
print("Kernel SVM")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)

# Algoritmo de clasificacion Naive Bayes
# Ajustar la clasificacion con el conjunto de entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[1, 0]
FN = conf_matrix[1, 1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
print("----------------------------------")
print("Naive Bayes")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)

# Algoritmo de clasificacion Decision Tree
# Ajustar la clasificacion con el conjunto de entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar resultados con matriz de confusion
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[1, 0]
FN = conf_matrix[1, 1]
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
print("----------------------------------")
print("Decision Tree")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)