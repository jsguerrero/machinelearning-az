# CLUSTERING KMEANS

# Importar el dataset
dataset <- read.csv('Mall_Customers.csv')
#dataset = dataset[, 2:3]
X <- dataset[, 4:5]

# METODO DEL CODO PARA SABER EL NUMERO OPTIMO DE CLUSTERS
set.seed(6)
wcss <- vector()
for(i in 1:10){
  wcss[i] <- sum(kmeans(X, i)$withinss)
}

plot(1:10,
     wcss,
     type = 'b',
     main = 'Metodo del codo',
     xlab = 'Numero de Clusters',
     ylab = 'WCSS(K)')

# APLICAR KMEANS
set.seed(29)
kmeans <- kmeans(X,
                 5,
                 iter.max = 300,
                 nstart = 10)

# VISUALIZACION DE CLUSTERS
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = FALSE,
         shade = TRUE,
         color = TRUE,
         #labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = 'Clustering de clientes',
         xlab = 'Ingresos anuales',
         ylab = 'Puntuacion')