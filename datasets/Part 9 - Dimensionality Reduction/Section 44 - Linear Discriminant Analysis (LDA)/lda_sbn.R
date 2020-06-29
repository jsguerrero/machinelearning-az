# LDA
# Importar dataset
dataset <- read.csv('Wine.csv')


# Dividir el dataset en entrenamiento y prueba
#install.packages('caTools')
library(caTools)
set.seed(123) #SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS
split <- sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
trainning_set <- subset(dataset, split == TRUE)
testing_set <- subset(dataset, split == FALSE)

# Escalado de variables
trainning_set[, -14] <- scale(trainning_set[, -14])
testing_set[, -14] <- scale(testing_set[, -14])

library(MASS)

lda <- lda(formula = Customer_Segment ~ .,
           data = trainning_set)

trainning_set <- as.data.frame(predict(lda, trainning_set))
trainning_set <- trainning_set[, c(5, 6, 1)]

testing_set <- as.data.frame(predict(lda, testing_set))
testing_set <- testing_set[, c(5, 6, 1)]

library(e1071)
# Ajustar svm con el conjunto de entrenamiento
classifier <- svm(formula = class ~ .,
                  data = trainning_set,
                  type = "C-classification",
                  kernel = "linear")

# Prediccion de la regresion logistica con el conjunto de prueba
y_pred <- predict(classifier,
                  newdata = testing_set[, -3])

# Evaluar resultados con matriz de confusion
conf_matrix <- table(testing_set[, 3], y_pred)

# Visualizacion de los resultados
#install.packages('ElemStatLearn')
library(ElemStatLearn)
#set <- trainning_set
set <- testing_set
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.03)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.03)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid <- predict(classifier,
                  newdata = grid_set)
plot(set[, -3],
     main = 'Clasificador (Conjunto de Entrenamiento)',
     xlab = 'DL1',
     ylab = 'DL2',
     xlim = range(X1),
     ylim = range(X2))
contour(X1,
        X2,
        matrix(as.numeric(y_grid),
               length(X1),
               length(X2)),
        add = TRUE)
points(grid_set,
       pch = '.',
       col = ifelse(y_grid == 2, 'deepskyblue',
                    ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set,
       pch = 21,
       bg = ifelse(set[, 3] == 2, 'blue3',
                   ifelse(set[, 3] == 1, 'green4', 'red3')))

#install.packages('ElemStatLearn')
library(ElemStatLearn)
set <- trainning_set
#set <- testing_set
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.02)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.02)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid <- predict(classifier,
                  newdata = grid_set)
plot(set[, -3],
     main = 'Clasificador (Conjunto de Prueba)',
     xlab = 'DL1',
     ylab = 'DL2',
     xlim = range(X1),
     ylim = range(X2))
contour(X1,
        X2,
        matrix(as.numeric(y_grid),
               length(X1),
               length(X2)),
        add = TRUE)
points(grid_set,
       pch = '.',
       col = ifelse(y_grid == 2, 'deepskyblue',
                    ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set,
       pch = 21,
       bg = ifelse(set[, 3] == 2, 'blue3',
                   ifelse(set[, 3] == 1, 'green4', 'red3')))