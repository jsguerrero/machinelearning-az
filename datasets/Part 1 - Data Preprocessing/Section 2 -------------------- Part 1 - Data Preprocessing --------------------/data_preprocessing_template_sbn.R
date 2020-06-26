# Plantilla de pre procesado
# Importar dataset
dataset <- read.csv('Data.csv')
#dataset = dataset[, 2:3]

# Dividir el dataset en entrenamiento y prueba
#install.packages('caTools')
library(caTools)
#set.seed(123) SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
trainning_set <- subset(dataset, split == TRUE)
testing_set <- subset(dataset, split == FALSE)

# Escalado de variables
# trainning_set[, 2:3] <- scale(trainning_set[, 2:3])
# testing_set[, 2:3] <- scale(testing_set[, 2:3])
