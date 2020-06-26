# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:02:49 2020

@author: jguerrero
"""

# CLUSTERING JERARQUICO

# SE IMPORTAN LAS LIBRERIAS PARA TRATAR LOS DATOS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SE IMPORTA EL ARCHIVO DE DATOS
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# METODO DEL DENDROGRAMA PARA SABER EL NUMERO OPTIMO DE CLUSTERS
import scipy.cluster.hierarchy as sch
dendrograma = sch.dendrogram(sch.linkage(X, method = 'ward', metric = 'euclidean'))
plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia euclidea')
plt.show()

# APLICAR CLUSTERING JERARQUICO
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# VISUALIZACION DE CLUSTERS
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
plt.title("Clusters de clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuacion de gastos")
plt.legend()
plt.show()
