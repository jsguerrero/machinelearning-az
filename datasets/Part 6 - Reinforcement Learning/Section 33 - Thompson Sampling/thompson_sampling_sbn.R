# MUESTREO THOMPSON

# Importar dataset
dataset <- read.csv('Ads_CTR_Optimisation.csv')

# ALGORITMO DE # MUESTREO THOMPSON
# N = NUMERO DE RONDAS O USUARIOS
# d = NUMERO DE OPCIONES O CLASES
N <- dim(dataset)[1]
d <- dim(dataset)[2]
# CONTADORES DE RECOMPENSAS POSITIVAS Y NEGATIVAS
count_positive_rewards <- integer(d)
count_negative_rewards <- integer(d)
# MEJORES OPCIONES SELECCIONADAS DE CADA RONDA
best_selected <- integer(0)
# RECOMPENSA TOTAL
total_reward <- 0

# SE RECORREN TODAS LAS RONDAS
for(n in 1:N){
  max_random <- 0
  best <- 0
  # SE GENERA LA DISTRIBUCION ALEATORIA BETA Y SE SELECCIONA EL DE VALOR MAYOR
  for(i in 1:d){
    random_beta <- rbeta(n = 1,
                         shape1 = count_positive_rewards[i] + 1,
                         shape2 = count_negative_rewards[i] + 1)
    if(random_beta > max_random){
      max_random <- random_beta
      best <- i
    }
  }
  # UNA VEZ QUE SE RECORREN TODAS LAS OPCIONES SE ACTUALIZAN
  # SE AGREGA LA OPCION CON EL MAXIMO LIMITE SUPERIOR
  best_selected <- append(best_selected, best)
  # SE SUMA LA RECOMPENSA OBTENIDA DE ESA OPCION DENTRO DE LA RONDA
  reward <- dataset[n, best]
  # SE ACTUALIZA LA CONTABILIDAD DE LAS RECOMPENSAS POSITIVAS Y NEGATIVAS
  if(reward == 1){
    count_positive_rewards[best] = count_positive_rewards[best] + 1
  }
  else{
    count_negative_rewards[best] = count_negative_rewards[best] + 1
  }
  total_reward <- total_reward + reward
}

# HISTOGRAMA DE RESULTADOS
hist(best_selected,
     col = 'lightblue',
     main = 'Histograma de anuncios',
     xlab = 'Id del anuncio',
     ylab = 'Frecuencia de visualización')