######################################################################################
#Machine Learning practice

#Objective: Create model to predict whether a cancer is malignant or benign from biopsy details

######################################################################################

#Import data 
library(mlbench)
data(BreastCancer)
dataset <- BreastCancer

######################################################################################
#EDA
######################################################################################


#Check out dim of dataset
dim(dataset)

#View dataset
View(dataset) #datasets looks to be full of nominal/ordinal variables

#Check out type of each attribute
sapply(dataset, class)

#Check out first 5 rows
head(dataset) 

#Check out bottom 5 rows
tail(dataset)

#Check out levels of column with malignant or benign to make sure those are the only things there
levels(dataset$Class)

#See what percentage of data is either m or b
percentage <- prop.table(table(dataset$Class)) * 100
cbind(freq = table(dataset$Clas), percentage = percentage)

#Check out structure of dataset
str(dataset)

#Get a summary of dataset
summary(dataset)

##########################################################################

#Create train and test set

############################################################################


library(caret)
validation_index <- createDataPartition(dataset$Class, p = .8, list = F)
test <- dataset[-validation_index,]
dataset <- dataset[validation_index,]

##########################################################################

#Create plots to visualize dataset and look for patterns

##########################################################################


#Split predictor and response variable of data (x = predictors, y = response (Class)) 
x = dataset[,2:10] #Didn't include ID since it is neither a predictor or response varible

#Split vector and factor attributes
xv = dataset[,2:5]
xf = dataset[,6:9]
y = dataset[,11] 

#Create univariate boxplot of numeric data to see how each attribute is structured
par(mfrow = c(1,4))
for(i in 2:5) {
  boxplot(x[,i], main = names(dataset)[i])
}

#Create histogram with stat = "count" to get a visual of factor attributes
install.packages("gridExtra")
library(ggplot2)
library(gridExtra)

plot_data_column = function(data, column)
  ggplot(data = x, aes_string(x = column)) + geom_histogram(stat = "count") + xlab(column)

myplots2 <- lapply(colnames(x), plot_data_column, data = x)
do.call(grid.arrange, c(myplots2, ncol = 3))


##Multivariate plots to see how attributes interact
for(i in 2:9){
  print(ggplot(dataset, aes(x= dataset[,i], fill = Class)) + geom_histogram(stat ="count") + facet_grid(.~Class) + xlab(names(dataset)[i]))
}

#Another Way to get previous chart (Multivariate)#

plot_multivariate = function(data, column)
  ggplot(data = dataset, aes_string(x= column)) + geom_histogram(stat ="count") + facet_grid(.~Class) + xlab(column)


multiplots <- lapply(colnames(dataset), plot_multivariate, data = dataset)
do.call(grid.arrange, c(multiplots, ncol = 3))

####################################################################################

#Evaluate some models against others (1. setup test harness to use 10 fold cross validation, 2. build 5 models to predict species from flower measurements, 3. select best model)

##Use 10 fold cross validtion to estimate accuracy
#Splits dataset into 10 parts, train in 9 and test in 1 and release for combinations of train-test splits. We will also repeat the process 3 times for each algorithm with diff split of the data into 10 groups, in an effort to get a more accurate estimate. 
control <- trainControl(method = "cv", number = 10)

#Using accuracy metric to judge how well our model works. Ratio of number of correctly predicted individed by the total number of instances then multipled by 100 to get a percentage
metric <- "Accuracy"


##################################################################################

# 
#Use a decision tree to create model to figure if tumor is beign or malgnant
library(rpart)
set.seed(123)

#Get correct syntax for variable names
colnames(dataset)

outcomeName <- 'Class'
predictors <- names(dataset)[!names(dataset) %in% outcomeName]

#Since we only want either 1 or zero (b or m) we will use method = class instead of annova which is for continuous variable. 
fit <- rpart(Class ~ Cl.thickness + Cell.size + Cell.shape + Marg.adhesion + Epith.c.size + Bare.nuclei + Bl.cromatin + Normal.nucleoli + Mitoses, 
              data = dataset,
              method = "class"
             )


#See decision tree model as plot

library(rattle)
library(rpart.plot)
library(RColorBrewer)
par(mfrow=c(1,1))

fancyRpartPlot(fit) ##Full Model has a lot of information. We need to prune to prevent from overfitting

#Check Accuracy of full model
pred <- predict(object = fit, test[,predictors], type = "class")
confusionMatrix(pred, test$Class) #Accuracy .9279 

#Prune
printcp(fit)

set.seed(123)

#Select trimming that has least error
ptree<- prune(fit, cp= fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

pred <- predict(object = ptree, test[,predictors], type = "class")
confusionMatrix(pred, test$Class) #Accuracy .9459

#Plot pt decision tree
fancyRpartPlot(ptree, uniform=TRUE,main="Pruned Classification Tree")




#_________________********* Other possible code for making model and checking accuracy

library(caret)
library(tidyverse)
library(dplyr)

set.seed(123)


df <- BreastCancer %>%
  select(-Id)

index <- createDataPartition(df$Class, p = 0.75, list = F)
trainSet <- df[ index,]
testSet <- df[-index,]

outcomeName <- 'Class'
predictors <- names(trainSet)[!names(trainSet) %in% outcomeName]

ctrl <- trainControl(method = 'cv', number = 10, repeats = 5, savePredictions = T, classProbs = T, allowParallel = T)

#create parallel to make my computer work harder if necessary (Not my code, heard from co worker that this might be neccasary)
if (require('parallel', quietly = T, warn.conflicts = F)) {
  ctrl$workers <- parallel:::detectCores()
  ctrl$computeFunction <- mclapply
  ctrl$computeArgs <- list(mc.preschedule = F, mc.set.seed = F)
}

#Create Decision Tree Model

fit <- train(trainSet[,predictors], trainSet[,outcomeName], method = 'rpart', trControl = ctrl, metric = 'Accuracy')

#Create fitted values from model
pred <- predict.train(object = fit, testSet[, predictors], type = 'raw')

#Check how accurate the model is
confusionMatrix(pred, testSet$Class) ##Accuracy of this model is .9483

