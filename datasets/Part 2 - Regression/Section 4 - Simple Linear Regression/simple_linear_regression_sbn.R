# Regresion lineal simple
# Importar dataset
dataset <- read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]

# Dividir el dataset en entrenamiento y prueba
#install.packages('caTools')
library(caTools)
#set.seed(123) SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS
split <- sample.split(dataset$Salary, SplitRatio = 2/3)
trainning_set <- subset(dataset, split == TRUE)
testing_set <- subset(dataset, split == FALSE)

# PARA REGRESION LINEAL SIMPLE NO SE REQUIERE ESCALAR DATOS
# Escalado de variables
# trainning_set[, 2:3] <- scale(trainning_set[, 2:3])
# testing_set[, 2:3] <- scale(testing_set[, 2:3])

# Crear modelo de regresion lineal simple con los datos de entrenamiento
regression <- lm(formula = Salary ~ YearsExperience,
                 data = trainning_set)

#summary(regression)

# Predecir el conjunto de prueba
# PARA USAR PREDICT LAS COLUMNAS DEL DATASET Y DE LOS NUEVOS DATOS DEBEN LLAMARSE IGUAL
y_pred <- predict(regression,
                  newdata = testing_set)

# Visualizar los resultados de entrenamiento
library(ggplot2)
ggplot() +
  geom_point(aes(x = trainning_set$YearsExperience,
                 y = trainning_set$Salary),
             colour = 'red') +
  geom_line(aes(x = trainning_set$YearsExperience,
                y = predict(regression,
                            newdata = trainning_set)),
            colour = 'blue') +
  ggtitle('Sueldo vs Años de experiencia [Conjunto de Entrenamiento]') +
  xlab('Años de experiencia') +
  ylab('Sueldo (USD $)')

# Visualizar los resultados de prueba
library(ggplot2)
ggplot() +
  geom_point(aes(x = testing_set$YearsExperience,
                 y = testing_set$Salary),
             colour = 'red') +
   geom_line(aes(x = trainning_set$YearsExperience,
                 y = predict(regression,
                             newdata = trainning_set)),
             colour = 'blue') +
  # geom_line(aes(x = testing_set$YearsExperience,
  #               y = y_pred),
  #           colour = 'blue') +
  ggtitle('Sueldo vs Años de experiencia [Conjunto de Test]') +
  xlab('Años de experiencia') +
  ylab('Sueldo (USD $)')