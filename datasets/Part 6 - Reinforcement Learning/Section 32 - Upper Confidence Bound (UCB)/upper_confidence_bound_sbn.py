# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:35:34 2020

@author: jguerrero
"""

# UPPER CONFIDENCE BOUND (UCB)
# SE IMPORTAN LAS LIBRERIAS PARA TRATAR LOS DATOS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SE IMPORTA EL ARCHIVO DE DATOS
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# ALGORITMO DE UPPER CONFIDENCE BOUND
import math
# N = NUMERO DE RONDAS O USUARIOS
# d = NUMERO DE OPCIONES O CLASES
N = dataset.shape[0]
d = dataset.shape[1]
# NUMERO DE SELECCIONES DE LAS OPCIONES
count_selections = [0] * d
# SUMA DE RECOMPENSAS OBTENIDAS
sum_rewards = [0] * d
# MEJORES OPCIONES SELECCIONADAS DE CADA RONDA
best_selected = []
# RECOMPENSA TOTAL
total_reward = 0

# SE RECORREN TODAS LAS RONDAS
for n in range(0, N):
    max_upper_bound = 0
    best = 0
    # SE CALCULA EL LIMITE SUPERIOR DEL INTERVALO DE CONFIANZA DE CADA OPCION
    for i in range(0, d):
        # ASEGURA QUE LA OPCION SE HA SELECCIONADO AL MENOS UNA VEZ
        if (count_selections[i] > 0):
            avg_rewards = sum_rewards[i] / count_selections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / count_selections[i])
            upper_bound = avg_rewards + delta_i
        else:
            # ESTE VALOR SE USA BASTANTE ALTO DEBIDO A LAS CONDICIONES INICIALES
            # DONDE NO SE PUEDEN DEFINIR LAS RECOMPENSAS SIN INFORMACIÓN PREVIA
            # POR ESO CON ESTE VALOR SE GARANTIZA QUE LAS PRIMERAS d RONDAS
            # SE CONSIDERE CADA OPCION COMO LA MEJOR
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            best = i

    # UNA VEZ QUE SE RECORREN TODAS LAS OPCIONES SE ACTUALIZAN
    # SE AGREGA LA OPCION CON EL MAXIMO LIMITE SUPERIOR
    best_selected.append(best)
    # SE AUMENTA EL CONTEO DE SELECCION DE LA OPCION
    count_selections[best] = count_selections[best] + 1
    # SE SUMA LA RECOMPENSA OBTENIDA DE ESA OPCION DENTRO DE LA RONDA
    reward = dataset.values[n, best]
    sum_rewards[best] = sum_rewards[best] + reward
    total_reward = total_reward + reward

# HISTOGRAMA DE RESULTADOS
plt.hist(best_selected)
plt.title('Histograma de anuncios')
plt.xlabel('Id del anuncio')
plt.ylabel('Frecuencia de visualización')
plt.show()