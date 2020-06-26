# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:08:41 2020

@author: jguerrero
"""
# CLUSTERING KMEANS

# SE IMPORTAN LAS LIBRERIAS PARA TRATAR LOS DATOS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SE IMPORTA EL ARCHIVO DE DATOS
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# METODO DEL CODO PARA SABER EL NUMERO OPTIMO DE CLUSTERS
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de Clusters")
plt.ylabel("WCSS(k)")
plt.show()

# APLICAR KMEANS
kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# VISUALIZACION DE CLUSTERS
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "yellow", label = "Baricentros")
plt.title("Clusters de clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuacion de gastos")
plt.legend()
plt.show()
