# ARBOLES DE DECISION
# Importar dataset
dataset <- read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Codificar la variable de clasificacion como factor
dataset$Purchased = factor(dataset$Purchased,
                           levels = c(0, 1))

# Dividir el dataset en entrenamiento y prueba
#install.packages('caTools')
library(caTools)
set.seed(123) #SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)
trainning_set <- subset(dataset, split == TRUE)
testing_set <- subset(dataset, split == FALSE)

# Ajustar la clasificacion con el conjunto de entrenamiento
library(rpart)
classifier <- rpart(formula = Purchased ~ .,
                   data = trainning_set)

# Prediccion de la clasificacion con el conjunto de prueba
y_pred <- predict(classifier,
                  newdata = testing_set[, -3],
                  type = 'class')

# Evaluar resultados con matriz de confusion
conf_matrix <- table(testing_set[, 3], y_pred)

# Visualizacion de los resultados
#install.packages('ElemStatLearn')
library(ElemStatLearn)
#set <- trainning_set
set <- testing_set
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 250)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid <- predict(classifier,
                  newdata = grid_set,
                  type = 'class')
plot(set[, -3],
     main = 'Clasificador (Conjunto de Entrenamiento)',
     xlab = 'Edad',
     ylab = 'Sueldo Estimado',
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
       col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set,
       pch = 21,
       bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# VISUALIZACION DE ARBOL DE DECISION
plot(classifier)
text(classifier)