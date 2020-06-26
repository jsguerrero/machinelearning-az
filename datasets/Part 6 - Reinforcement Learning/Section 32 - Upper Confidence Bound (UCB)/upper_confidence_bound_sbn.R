# UPPER CONFIDENCE BOUND (UCB)

# Importar dataset
dataset <- read.csv('Ads_CTR_Optimisation.csv')

# ALGORITMO DE UPPER CONFIDENCE BOUND
# N = NUMERO DE RONDAS O USUARIOS
# d = NUMERO DE OPCIONES O CLASES
N <- dim(dataset)[1]
d <- dim(dataset)[2]
# NUMERO DE SELECCIONES DE LAS OPCIONES
count_selections <- integer(d)
# SUMA DE RECOMPENSAS OBTENIDAS
sum_rewards <- integer(d)
# MEJORES OPCIONES SELECCIONADAS DE CADA RONDA
best_selected <- integer(0)
# RECOMPENSA TOTAL
total_reward <- 0

# SE RECORREN TODAS LAS RONDAS
for(n in 1:N){
  max_upper_bound <- 0
  best <- 0
  # SE CALCULA EL LIMITE SUPERIOR DEL INTERVALO DE CONFIANZA DE CADA OPCION
  for(i in 1:d){
    # ASEGURA QUE LA OPCION SE HA SELECCIONADO AL MENOS UNA VEZ
    if (count_selections[i] > 0){
      avg_rewards <- sum_rewards[i] / count_selections[i]
      delta_i <- sqrt(3 / 2 * log(n + 1) / count_selections[i])
      upper_bound <- avg_rewards + delta_i
    }
    else{
      # ESTE VALOR SE USA BASTANTE ALTO DEBIDO A LAS CONDICIONES INICIALES
      # DONDE NO SE PUEDEN DEFINIR LAS RECOMPENSAS SIN INFORMACIÓN PREVIA
      # POR ESO CON ESTE VALOR SE GARANTIZA QUE LAS PRIMERAS d RONDAS
      # SE CONSIDERE CADA OPCION COMO LA MEJOR
      upper_bound <- 1e400
    }
    if(upper_bound > max_upper_bound){
      max_upper_bound <- upper_bound
      best <- i
    }
  }
  # UNA VEZ QUE SE RECORREN TODAS LAS OPCIONES SE ACTUALIZAN
  # SE AGREGA LA OPCION CON EL MAXIMO LIMITE SUPERIOR
  best_selected <- append(best_selected, best)
  # SE AUMENTA EL CONTEO DE SELECCION DE LA OPCION
  count_selections[best] <- count_selections[best] + 1
  # SE SUMA LA RECOMPENSA OBTENIDA DE ESA OPCION DENTRO DE LA RONDA
  reward <- dataset[n, best]
  sum_rewards[best] <- sum_rewards[best] + reward
  total_reward <- total_reward + reward
}

# HISTOGRAMA DE RESULTADOS
hist(best_selected,
     col = 'lightblue',
     main = 'Histograma de anuncios',
     xlab = 'Id del anuncio',
     ylab = 'Frecuencia de visualización')