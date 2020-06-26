# CLUSTERING JERARQUICO

# Importar el dataset
dataset <- read.csv('Mall_Customers.csv')
#dataset = dataset[, 2:3]
X <- dataset[, 4:5]

# METODO DEL DENDROGRAMA PARA SABER EL NUMERO OPTIMO DE CLUSTERS
dendrograma <- hclust(dist(X, method = 'euclidean'),
                     method = 'ward.D')

plot(dendrograma,
     main = 'Dendrograma',
     xlab = 'Numero de Clusters',
     ylab = 'Distrancia Euclidea')

# APLICAR CLUSTERING JERARQUICO
hc <- hclust(dist(X, method = 'euclidean'),
                     method = 'ward.D')

y_hc <- cutree(hc, k = 5)

# VISUALIZACION DE CLUSTERS
library(cluster)
clusplot(X,
         y_hc,
         lines = FALSE,
         shade = TRUE,
         color = TRUE,
         #labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = 'Clustering de clientes',
         xlab = 'Ingresos anuales',
         ylab = 'Puntuacion')