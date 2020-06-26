# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:33:07 2020

@author: jguerrero
"""

# MUESTREO THOMPSON
# SE IMPORTAN LAS LIBRERIAS PARA TRATAR LOS DATOS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SE IMPORTA EL ARCHIVO DE DATOS
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# ALGORITMO DE MUESTREO THOMPSON
import random
# N = NUMERO DE RONDAS O USUARIOS
# d = NUMERO DE OPCIONES O CLASES
N = dataset.shape[0]
d = dataset.shape[1]
# CONTADORES DE RECOMPENSAS POSITIVAS Y NEGATIVAS
count_positive_rewards = [0] * d
count_negative_rewards = [0] * d
# MEJORES OPCIONES SELECCIONADAS DE CADA RONDA
best_selected = []
# RECOMPENSA TOTAL
total_reward = 0

# SE RECORREN TODAS LAS RONDAS
for n in range(0, N):
    max_random = 0
    best = 0
    # SE GENERA LA DISTRIBUCION ALEATORIA BETA Y SE SELECCIONA EL DE VALOR MAYOR
    for i in range(0, d):
        random_beta = random.betavariate(count_positive_rewards[i] + 1, count_negative_rewards[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            best = i
    # UNA VEZ QUE SE RECORREN TODAS LAS OPCIONES SE ACTUALIZAN
    # SE AGREGA LA OPCION CON EL MAXIMO VALOR ALEATORIO BETA
    best_selected.append(best)
    # SE SUMA LA RECOMPENSA OBTENIDA DE ESA OPCION DENTRO DE LA RONDA
    reward = dataset.values[n, best]
    # SE ACTUALIZA LA CONTABILIDAD DE LAS RECOMPENSAS POSITIVAS Y NEGATIVAS
    if reward == 1:
        count_positive_rewards[best] = count_positive_rewards[best] + 1
    else:
        count_negative_rewards[best] = count_negative_rewards[best] + 1
    # RECOMPENSA TOTAL
    total_reward = total_reward + reward

# HISTOGRAMA DE RESULTADOS
plt.hist(best_selected)
plt.title('Histograma de anuncios')
plt.xlabel('Id del anuncio')
plt.ylabel('Frecuencia de visualización')
plt.show()