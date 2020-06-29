# REDES NEURONALES ARTIFICIALES

# PARTE 1 - PRE PROCESADO DE DATOS
# DATASET
dataset <- read.csv('Churn_Modelling.csv')
dataset = dataset[, 4:14]

# CODIFICAR VARIABLES CATEGORICAS COMO FACTOR
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c("France", "Spain", "Germany"),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                      levels = c("Female", "Male"),
                                      labels = c(1, 2)))

# ESCALADO DE VARIABLES
dataset[, -dim(dataset)[2]] <- scale(dataset[, -dim(dataset)[2]])

# DIVIDIR EL DATASET EN ENTRENAMIENTO Y PRUEBA
#install.packages('caTools')
library(caTools)
set.seed(123) #SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS
split <- sample.split(dataset$Exited, SplitRatio = 0.8)
trainning_set <- subset(dataset, split == TRUE)
testing_set <- subset(dataset, split == FALSE)

# PARTE 2 - ALGORITMO DE RED NEURONAL ARTIFICIAL
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
classifier <- h2o.deeplearning(y = "Exited",
                               training_frame = as.h2o(trainning_set),
                               activation = "Rectifier",
                               hidden = c(6,6),
                               epochs = 100,
                               train_samples_per_iteration = -2)

# PARTE 3 - CALCULAR PREDICCIONES FINALES Y EVALUAR EL MODELO
# PREDICCION DE LA CLASIFICACION CON EL CONJUNTO DE PRUEBA
y_pred <- h2o.predict(classifier,
                      newdata = as.h2o(testing_set[, -11]))

# CONVERTIR DE LA PROPABILIDAD A UNA RESPUESTA
y_pred <- (y_pred > 0.5)

y_pred <- as.vector(y_pred)

# Evaluar resultados con matriz de confusion
conf_matrix <- table(testing_set[, 11], y_pred)

# CERRAR SESION H2O
h2o.shutdown()
