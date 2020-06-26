# REGLAS ECLAT

# Importar el dataset
#dataset <- read.csv('Market_Basket_Optimisation.csv', header = FALSE)
#dataset = dataset[, 2:3]
#install.packages('arules')
library(arules)
dataset <- read.transactions('Market_Basket_Optimisation.csv',
                             sep = ',',
                             rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# ENTRENAR ALGORITMO ECLAT
# SOPORTE PRODUCTOS CON VENTA MINIMO DE 3 VECES POR SEMANA DENTRO DEL TOTAL DE LOS DATOS
# 3 * 7 / 7500 = 0.0028
rules <- eclat(data = dataset,
                 parameter = list(support = 0.004,
                                  minlen = 2))

# VISUALIZACION DE LAS REGLAS
inspect(sort(rules, by = 'support')[1:10])