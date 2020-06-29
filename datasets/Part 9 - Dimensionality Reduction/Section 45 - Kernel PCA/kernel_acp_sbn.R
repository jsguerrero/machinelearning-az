# KERNEL PCA
# Importar dataset
dataset <- read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Dividir el dataset en entrenamiento y prueba
#install.packages('caTools')
library(caTools)
set.seed(123) #SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)
trainning_set <- subset(dataset, split == TRUE)
testing_set <- subset(dataset, split == FALSE)

# Escalado de variables
trainning_set[, 1:2] <- scale(trainning_set[, 1:2])
testing_set[, 1:2] <- scale(testing_set[, 1:2])

#install.packages('kernlab')
library(kernlab)

kpca <- kpca(~., data = trainning_set[, -3], kernel = 'rbfdot', features = 2)

trainning_set_pca <- as.data.frame(predict(kpca, trainning_set))
trainning_set_pca$Purchased = trainning_set$Purchased

testing_set_pca <- as.data.frame(predict(kpca, testing_set))
testing_set_pca$Purchased = testing_set$Purchased

# Ajustar la regresion logistica con el conjunto de entrenamiento
classifier <- glm(formula = Purchased ~ .,
                  data = trainning_set_pca,
                  family = binomial)

# Prediccion de la regresion logistica con el conjunto de prueba
prob_pred <- predict(classifier,
                     type = 'response',
                     newdata = testing_set_pca[, -3])

y_pred <- ifelse(prob_pred > 0.5, 1, 0)

# Evaluar resultados con matriz de confusion
conf_matrix <- table(testing_set[, 3], y_pred)

# Visualizacion de los resultados
#install.packages('ElemStatLearn')
library(ElemStatLearn)
#set <- trainning_set
set <- testing_set_pca
X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.03)
X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.03)
grid_set <- expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set <- predict(classifier,
                    type = 'response',
                    newdata = grid_set)
y_grid <- ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Clasificador (Conjunto de Entrenamiento)',
     xlab = 'CP1',
     ylab = 'CP2',
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