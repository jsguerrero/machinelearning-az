# Regresion con bosques aleatrorios
# Importar dataset
dataset <- read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]
class(dataset[1])
class(dataset$Level)

# Dividir el dataset en entrenamiento y prueba
#install.packages('caTools')
# library(caTools)
# set.seed(123) SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS
# split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
# trainning_set <- subset(dataset, split == TRUE)
# testing_set <- subset(dataset, split == FALSE)

# Escalado de variables
# trainning_set[, 2:3] <- scale(trainning_set[, 2:3])
# testing_set[, 2:3] <- scale(testing_set[, 2:3])+

set.seed(1234)

# Ajustar la regresion de bosques aleatrorios al dataset
# install.packages('randomForest')
library(randomForest)
regression <- randomForest(x = dataset[1],
                           y = dataset$Salary,
                           ntree = 500)
#summary(regression)

# Prediccion de la regresion de bosques aleatrorios
y_pred <- predict(regression,
                  newdata = data.frame(Level = 6.5))

# Visualizacion de los resultados
library(ggplot2)
x_grid <- seq(min(dataset$Level), max(dataset$Level), 0.01)
#x_grid <- dataset$Level
ggplot() +
  geom_point(aes(x = dataset$Level,
                 y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid,
                y = predict(regression,
                            newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Modelo de Regresion') +
  xlab('Posición del empleado') +
  ylab('Sueldo en USD $')
