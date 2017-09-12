### This is the final project of Coursera DataScience Class8 Machine Learning
### The purpose is to use machine learning to evaluate how well the exercies is
### Charles 09/05/2017

# Set the environment
setwd("C:/Study/Coursera/1 Data-Science/2 RStudio/8 Class 8/Coursera_DataScience_Class8_MachineLearning")

library(caret)
library(ggplot2)
library(dplyr)
library(grid)
library(gridExtra)
library(manipulate)
library(olsrr)
library(rattle)
library(rpart.plot)
library(randomForest)

# Load the data
if (!file.exists("./Data")){
        dir.create("./Data")
}
urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(urlTrain, destfile = "./Data/trainFP")
download.file(urlTest, destfile = "./Data/testFP")
training <- read.csv("./Data/trainFP")
testing <- read.csv("./Data/testFP")
str(training)
head(training)

# Clean the data
near0VarInfo <- nearZeroVar(training, saveMetrics = TRUE)
near0Var <- nearZeroVar(training)
training <- training[,-near0Var]

processData <- function(data){
        indexKeep <- !sapply(data, function(x) any(is.na(x)))
        data <- data[,indexKeep]
        indexKeep <- !sapply(data, function(x) any(x == ""))
        data <- data[,indexKeep]
        colNames <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
                      "cvtd_timestamp", "new_window", "num_window")
        indexCol <- which(names(data) %in% colNames)
        data <- data[,-indexCol]
        return(data)
}

training <- processData(training)
dim(training)

# Do the same to test data set
near0Var2 <- nearZeroVar(testing)
testing <- testing[,-near0Var2]
testing <- processData(testing)
dim(testing)

# Test
table(complete.cases(training))
table(sapply(training[1,], class))

### Fit model
# Data Partitioning
set.seed(20170917)
inTrain <- createDataPartition(training$classe, p=0.8, list = FALSE)
newTraining <- training[inTrain,]
newTesting <- training[-inTrain,]
dim(newTraining)
dim(newTesting)

# Random Forest
modFitRF <- randomForest(classe ~., data = newTraining, ntree = 500)
predRF <- predict(modFitRF, newTesting)
conMatrix1 <- confusionMatrix(predRF, newTesting$classe)
conMatrix1$overall

# Test model
varImp(modFitRF)
varImpPlot(modFitRF)

# Train model in caret package
modFitRF2 <- train(classe ~., data = newTraining, method = "rf", trControl = trainControl(method = "cv",5), ntree=500)
predRF2 <- predict(modFitRF2, newTesting)
conMatrix2 <- confusionMatrix(predRF2, newTesting$classe)
conMatrix2$overall

# Tree visualization
treeModel <- rpart(classe ~ ., data=newTraining, method="class")
prp(treeModel)

# System information
Sys.info()[1:2]
R.version.string

























