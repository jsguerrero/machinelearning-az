# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:03:05 2020

@author: jguerrero
"""

# REGLAS APRIORI
# SE IMPORTAN LAS LIBRERIAS PARA TRATAR LOS DATOS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SE IMPORTA EL ARCHIVO DE DATOS
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, dataset.shape[1])])

# ENTRENAR ALGORITMO APRIORI
# SOPORTE PRODUCTOS CON VENTA MINIMO DE 3 VECES POR SEMANA DENTRO DEL TOTAL DE LOS DATOS
# 3 * 7 / 7500 = 0.0028
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# VISUALIZACION DE LAS REGLAS
result = list(rules)

result.sort(key=lambda x:x.ordered_statistics[0].lift, reverse=True)

result[0]