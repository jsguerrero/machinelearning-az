# Regresion lineal multiple
# Importar dataset
dataset <- read.csv('50_Startups.csv')
#dataset = dataset[, 2:3]

# Codificar datos categoricos
dataset$State <- factor(dataset$State,
                          levels = c('New York', 'California', 'Florida'),
                          labels = c(1, 2, 3))

# Dividir el dataset en entrenamiento y prueba
#install.packages('caTools')
library(caTools)
set.seed(123) #SOLO ES PARA QUE SIEMPRE SELECCIONE LOS MISMOS ELEMENTOS
split <- sample.split(dataset$Profit, SplitRatio = 0.8)
trainning_set <- subset(dataset, split == TRUE)
testing_set <- subset(dataset, split == FALSE)

# Escalado de variables
# trainning_set[, 2:3] <- scale(trainning_set[, 2:3])
# testing_set[, 2:3] <- scale(testing_set[, 2:3])

# Ajustar el modelo de regresion lineal multiple con los datos de entrenamiento
regression <- lm(formula = Profit ~ .,
                 data = trainning_set)

#summary(regression)

# Predecir el conjunto de prueba
y_pred <- predict(regression,
                  newdata = testing_set)

# Construir modelo optimo utilizando la Eliminacion hacia atras
SL <- 0.05
regression <- lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                 data = dataset)
summary(regression)

regression <- lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                 data = dataset)
summary(regression)

regression <- lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                 data = dataset)
summary(regression)

regression <- lm(formula = Profit ~ R.D.Spend,
                 data = dataset)
summary(regression)

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  print(summary(regressor))
  #return(summary(regressor))
  #return(regressor)
  return(x)
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
x_modeled <- backwardElimination(dataset, SL)

backwardElimination_rsquared <- function(x, sl){
  numVars = length(x)
  temp = matrix(data = 0, ncol = 5, nrow = 50)
  temp <- as.data.frame(temp)
  colnames(temp) <- colnames(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    adjR_before = summary(regressor)$adj.r.squared
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      temp[, j] = x[, j]
      x = x[, -j]
      tmp_regressor = lm(formula = Profit ~ ., data = x)
      adjR_after = summary(tmp_regressor)$adj.r.squared
      if (adjR_before >= adjR_after){
        x_rollback <- cbind(x[, -j], temp[,j], x[,numVars - 1])
        colnames(x_rollback) <- c(names(x)[-j], names(temp)[j], names(x)[length(x)])
        x <-as.data.frame(x_rollback)
      }
      else{
        numVars = numVars - 1
      }
    }
  }
  print(summary(regressor))
  #return(summary(regressor))
  #return(regressor)
  return(x)
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
x_modeled <- backwardElimination_rsquared(dataset, SL)