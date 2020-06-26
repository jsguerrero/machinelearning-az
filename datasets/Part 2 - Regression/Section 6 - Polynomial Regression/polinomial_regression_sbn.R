# Regresion polinomica
# Importar dataset
dataset <- read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

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

# Ajustar la regresion lineal con el dataset
lineal_reg <- lm(formula = Salary ~ .,
                 data = dataset)
# summary(lineal_reg)

# Ajustar la regresion polinomica con el dataset
dataset$Level2 <- dataset$Level ^ 2
dataset$Level3 <- dataset$Level ^ 3
dataset$Level4 <- dataset$Level ^ 4
poly_reg <- lm(formula = Salary ~ .,
                 data = dataset)
# summary(poly_reg)

# Visualizacion de los resultados del modelo lineal
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level,
                 y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level,
                y = predict(lineal_reg,
                            newdata = dataset)),
            colour = 'blue') +
  ggtitle('Modelo de Regresion Lineal') +
  xlab('Posición del empleado') +
  ylab('Sueldo en USD $')

# Visualizacion de los resultados del modelo polinomico
x_grid <- seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level,
                 y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid,
                y = predict(poly_reg,
                            newdata = data.frame(Level = x_grid,
                                                 Level2 = x_grid ^ 2,
                                                 Level3 = x_grid ^ 3,
                                                 Level4 = x_grid ^ 4))),
            colour = 'blue') +
  ggtitle('Modelo de Regresion') +
  xlab('Posición del empleado') +
  ylab('Sueldo en USD $')

# Prediccion de nuestros modelos
y_pred <- predict(lineal_reg,
                   newdata = data.frame(Level = 6.5))

y_pred <- predict(poly_reg,
                  newdata = data.frame(Level = 6.5,
                                       Level2 = 6.5 ^ 2,
                                       Level3 = 6.5 ^ 3,
                                       Level4 = 6.5 ^ 4))