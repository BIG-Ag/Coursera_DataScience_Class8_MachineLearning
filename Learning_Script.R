### This is the learning script of Coursera DataScience Machine Learning Class
### Charles 08/22/2017

# Set working directory and environment
setwd("C:/Study/Coursera/1 Data-Science/2 RStudio/8 Class 8/Coursera_DataScience_Class8_MachineLearning")

library(caret)
library(kernlab)
library(ISLR)
library(UsingR)
library(ggplot2)
library(dplyr)
library(grid)
library(gridExtra)
library(manipulate)
library(olsrr)
library(rattle)
library(rpart.plot)

##################################### Week 2 #####################
### Data Slicing
data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)

# K fold
set.seed(32323)
folds <- createFolds(y=spam$type,k=10,
                     list=TRUE,returnTrain=TRUE)
sapply(folds,length)
folds[[1]][1:10]

# Resample
set.seed(32323)
folds <- createResample(y=spam$type,times=10,
                        list=TRUE)
sapply(folds,length)
folds[[1]][1:10]

# Time slicing
set.seed(32323)
tme <- 1:1000
folds <- createTimeSlices(y=tme,initialWindow=20,
                          horizon=10)
names(folds)
folds$train[[1]]
folds$test[[1]]

### Traning Options
modelFit <- train(type ~.,data=training, method="glm")

args(train.default)
args(trainControl)


### Plotting predictors
data(Wage)
summary(Wage)
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)

# Feature Plot
featurePlot(x=training[,c("age","education","jobclass")],
            y = training$wage,
            plot="pairs")
# qplot
qplot(age,wage,data=training)
qplot(age,wage,colour=jobclass,data=training)
qq <- qplot(age,wage,colour=education,data=training)
qq +  geom_smooth(method='lm',formula=y~x)

# Cut2
cutWage <- cut2(training$wage,g=3)
table(cutWage)
p1 <- qplot(cutWage,age, data=training,fill=cutWage,
            geom=c("boxplot"))
p1

p2 <- qplot(cutWage,age, data=training,fill=cutWage,
            geom=c("boxplot","jitter"))
grid.arrange(p1,p2,ncol=2)

# Table
t1 <- table(cutWage,training$jobclass)
t1
prop.table(t1,1)

# Density plot
qplot(wage,colour=education,data=training,geom="density")

### Preprocessing
data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
hist(training$capitalAve,main="",xlab="ave. capital run length")

mean(training$capitalAve)
sd(training$capitalAve) # Too wide

# Standardizing
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve  - mean(trainCapAve))/sd(trainCapAve) 
mean(trainCapAveS)
sd(trainCapAveS)

# preProcess function
preObj <- preProcess(training[,-58],method=c("center","scale"))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
mean(trainCapAveS)
sd(trainCapAveS)

# in argument
set.seed(32343)
modelFit <- train(type ~.,data=training,
                  preProcess=c("center","scale"),method="glm")
modelFit

# BoxCox transform
preObj <- preProcess(training[,-58],method=c("BoxCox"))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
par(mfrow=c(1,2)); hist(trainCapAveS); qqnorm(trainCapAveS)

# Imputing data
set.seed(13343)
# Make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1],size=1,prob=0.05)==1
training$capAve[selectNA] <- NA
# Impute and standardize
preObj <- preProcess(training[,-58],method="knnImpute")
capAve <- predict(preObj,training[,-58])$capAve
# Standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth-mean(capAveTruth))/sd(capAveTruth)

quantile(capAve - capAveTruth)
quantile((capAve - capAveTruth)[selectNA])
quantile((capAve - capAveTruth)[!selectNA])


### Covariate Creation
data(Wage);
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]

table(training$jobclass)
dummies <- dummyVars(wage ~ jobclass,data=training)
head(predict(dummies,newdata=training))

# near zero variables
nsv <- nearZeroVar(training,saveMetrics=TRUE)
nsv

# Spline
library(splines)
bsBasis <- bs(training$age,df=3) 
bsBasis
# Curves with splines
lm1 <- lm(wage ~ bsBasis,data=training)
plot(training$age,training$wage,pch=19,cex=0.5)
points(training$age,predict(lm1,newdata=training),col="red",pch=19,cex=0.5)

predict(bsBasis,age=testing$age)


### PCA
# Correlated predictors
data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

M <- abs(cor(training[,-58]))
diag(M) <- 0
which(M > 0.8,arr.ind=T)

# plot correlated
names(spam)[c(34,32)]
plot(spam[,34],spam[,32])

# Rotate plot
X <- 0.71*training$num415 + 0.71*training$num857
Y <- 0.71*training$num415 - 0.71*training$num857
plot(X,Y)

# PCA
smallSpam <- spam[,c(34,32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[,1],prComp$x[,2])
# Check rotation
prComp$rotation

# PCA on all data
typeColor <- ((spam$type=="spam")*1 + 1)
prComp <- prcomp(log10(spam[,-58]+1))
plot(prComp$x[,1],prComp$x[,2],col=typeColor,xlab="PC1",ylab="PC2")

# PCA on caret
preProc <- preProcess(log10(spam[,-58]+1),method="pca",pcaComp=2)
spamPC <- predict(preProc,log10(spam[,-58]+1))
plot(spamPC[,1],spamPC[,2],col=typeColor)

# Use training dataset    confuseMatrix
preProc <- preProcess(log10(training[,-58]+1),method="pca",pcaComp=2)
trainPC <- predict(preProc,log10(training[,-58]+1))
modelFit <- train(training$type ~ . ,method="glm",data=trainPC)

testPC <- predict(preProc,log10(testing[,-58]+1))
confusionMatrix(testing$type,predict(modelFit,testPC))

# Alternative way
modelFit <- train(training$type ~ .,method="glm",preProcess="pca",data=training)
confusionMatrix(testing$type,predict(modelFit,testing))



### Predict with regression
# Old faithful eruptions
data(faithful); set.seed(333)
inTrain <- createDataPartition(y=faithful$waiting,
                               p=0.5, list=FALSE)
trainFaith <- faithful[inTrain,]; testFaith <- faithful[-inTrain,]
head(trainFaith)

plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
# Fit a linear model
lm1 <- lm(eruptions ~ waiting,data=trainFaith)
summary(lm1)
# Plot a line
plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
lines(trainFaith$waiting,lm1$fitted,lwd=3)
# Predict
coef(lm1)[1] + coef(lm1)[2]*80
newdata <- data.frame(waiting=80)
predict(lm1,newdata)
# Test set
par(mfrow=c(1,2))
plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
lines(trainFaith$waiting,predict(lm1),lwd=3)
plot(testFaith$waiting,testFaith$eruptions,pch=19,col="blue",xlab="Waiting",ylab="Duration")
lines(testFaith$waiting,predict(lm1,newdata=testFaith),lwd=3)

# Test errors
# Calculate RMSE on training
sqrt(sum((lm1$fitted-trainFaith$eruptions)^2))

# Calculate RMSE on test
sqrt(sum((predict(lm1,newdata=testFaith)-testFaith$eruptions)^2))

# Prediction interval
pred1 <- predict(lm1,newdata=testFaith,interval="prediction")
ord <- order(testFaith$waiting)
plot(testFaith$waiting,testFaith$eruptions,pch=19,col="blue")
matlines(testFaith$waiting[ord],pred1[ord,],type="l",,col=c(1,2,2),lty = c(1,1,1), lwd=3)

# Use Caret package
modFit <- train(eruptions ~ waiting,data=trainFaith,method="lm")
summary(modFit$finalModel)



### Prediction regression Multivariable Covariates
data(Wage); Wage <- subset(Wage,select=-c(logwage))
summary(Wage)

inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]
dim(training); dim(testing)

# Feature plot
featurePlot(x=training[,c("age","education","jobclass")],
            y = training$wage,
            plot="pairs")
# age to wage
qplot(age,wage,data=training)
qplot(age,wage,colour=jobclass,data=training)
qplot(age,wage,colour=education,data=training)

# Fit a linear model
modFit<- train(wage ~ age + jobclass + education,
               method = "lm",data=training)
finMod <- modFit$finalModel
print(modFit)
# Plot residuals
plot(finMod,1,pch=19,cex=0.5,col="#00000010")
qplot(finMod$fitted,finMod$residuals,colour=race,data=training)
plot(finMod$residuals,pch=19)
# Compare test set
pred <- predict(modFit, testing)
qplot(wage,pred,colour=year,data=testing)
# Fit all
modFitAll<- train(wage ~ .,data=training,method="lm")
pred <- predict(modFitAll, testing)
qplot(wage,pred,data=testing)


################ Quiz 2
# Q1
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
str(predictors)
str(diagnosis)
adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

adData = data.frame(predictors)
trainIndex = createDataPartition(diagnosis,p=0.5,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

# Q2
data(concrete)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

names(mixtures)
names(concrete)
nrow(training)

training <- mutate(training, index = 1:nrow(training))
head(training)
cutIndex <- cut2(training$index, g=10)
breaks <- 10
qplot(index, CompressiveStrength, data = training, color = cut2(training$Cement, g=breaks))
qplot(index, CompressiveStrength, data = training, color = cut2(training$BlastFurnaceSlag, g=breaks))
qplot(index, CompressiveStrength, data = training, color = cut2(training$FlyAsh, g=breaks))
qplot(index, CompressiveStrength, data = training, color = cut2(training$Water, g=breaks))

# Q3
data(concrete)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
hist(training$Superplasticizer, breaks = 20)
hist(log(training$Superplasticizer +1), breaks = 20)

# Q4
set.seed(3433); data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

preProc <- preProcess(log10(training[,-58]+1),method="pca",pcaComp=2)
head(adData)
grep("^[Ii][Ll].*", names(adData))
adIL <- training[,grep("^[Ii][Ll].*", names(adData))]
head(adIL)
preObj <- preProcess(adIL, method = c("center","scale","pca"), thresh = 0.8)
preObj

# Q5
set.seed(3433);data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

trainIL <- data.frame(training[,grep("^[Ii][Ll].*", names(adData))], training$diagnosis)
testIL <- data.frame(testing[,grep("^[Ii][Ll].*", names(adData))], testing$diagnosis)

NoPCFit <- train(training.diagnosis ~ ., data = trainIL, method = "glm")
NoPCFitResult <- confusionMatrix(testIL[,13], predict(NoPCFit, testIL[,-13]))
NoPCFitResult

preObj <- preProcess(trainIL[,-13], method ="pca", thresh = 0.8)
trainPC <- predict(preObj, trainIL[-13])
testPC <- predict(preObj, testIL[-13])
head(trainPC)
# Add a new column
trainPC <- data.frame(trainPC, training$diagnosis)

PCFit <- train(training.diagnosis ~ ., data = trainPC, method = "glm")
PCPredict <- predict(PCFit, testPC)
PCFitResult <- confusionMatrix(testing$diagnosis, PCPredict)
PCFitResult

# Alternative way
PCFit <- train(training.diagnosis ~.,
               data = trainIL, 
               method ="glm",
               preProc = "pca",
               trControl = trainControl(preProcOptions = list(thresh = 0.8)))
PCFitResult <- confusionMatrix(testIL[,13], predict(PCFit,testIL[,-13]))
PCFitResult





#################################### Week 3
### Prediction with trees
data(iris)
names(iris)
table(iris$Species)
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)
qplot(Petal.Width,Sepal.Width,colour=Species,data=training)
# Train model
modFit <- train(Species ~ .,method="rpart",data=training)
print(modFit$finalModel)
# Plot
plot(modFit$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)
# Rattle package       Not support on this version of R
library(rattle)
fancyRpartPlot(modFit$finalModel)
# Predict
predict(modFit,newdata=testing)


### Bagging
library(ElemStatLearn)
data(ozone,package="ElemStatLearn")
ozone <- ozone[order(ozone$ozone),]
head(ozone)
# Resample
ll <- matrix(NA,nrow=10,ncol=155)
for(i in 1:10){
        ss <- sample(1:dim(ozone)[1],replace=T)
        ozone0 <- ozone[ss,]; ozone0 <- ozone0[order(ozone0$ozone),]
        loess0 <- loess(temperature ~ ozone,data=ozone0,span=0.2)
        ll[i,] <- predict(loess0,newdata=data.frame(ozone=1:155))
}
# Plot
plot(ozone$ozone,ozone$temperature,pch=19,cex=0.5)
for(i in 1:10){lines(1:155,ll[i,],col="grey",lwd=2)}
lines(1:155,apply(ll,2,mean),col="red",lwd=2)

# More bagging in caret
predictors = data.frame(ozone=ozone$ozone)
temperature = ozone$temperature
treebag <- bag(predictors, temperature, B = 10,
               bagControl = bagControl(fit = ctreeBag$fit,
                                       predict = ctreeBag$pred,
                                       aggregate = ctreeBag$aggregate))
# Custom bagging
plot(ozone$ozone,temperature,col='lightgrey',pch=19)
points(ozone$ozone,predict(treebag$fits[[1]]$fit,predictors),pch=19,col="red")
points(ozone$ozone,predict(treebag,predictors),pch=19,col="blue")


### Random forests
data(iris)
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

modFit <- train(Species~ .,data=training,method="rf",prox=TRUE)
modFit
# Get a single tree
getTree(modFit$finalModel,k=2)
# Class centers
irisP <- classCenter(training[,c(3,4)], training$Species, modFit$finalModel$prox)
irisP <- as.data.frame(irisP); irisP$Species <- rownames(irisP)
p <- qplot(Petal.Width, Petal.Length, col=Species,data=training)
p + geom_point(aes(x=Petal.Width,y=Petal.Length,col=Species),size=5,shape=4,data=irisP)
# Predict new  values
pred <- predict(modFit,testing); testing$predRight <- pred==testing$Species
table(pred,testing$Species)
qplot(Petal.Width,Petal.Length,colour=predRight,data=testing,main="newdata Predictions")




########## Boosting
library(ISLR)
data(Wage)
Wage <- subset(Wage,select=-c(logwage))
inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]
# Fit a model          # gbm boosting with trees
modFit <- train(wage ~ ., method="gbm",data=training,verbose=FALSE)
print(modFit)
qplot(predict(modFit,testing),wage,data=testing)



######## Model based Prediction
names(iris)
table(iris$Species)
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)
modlda = train(Species ~ .,data=training,method="lda")
modnb = train(Species ~ ., data=training,method="nb")
plda = predict(modlda,testing); pnb = predict(modnb,testing)
table(plda,pnb)
equalPredictions = (plda==pnb)
qplot(Petal.Width,Sepal.Width,colour=equalPredictions,data=testing)


########## Quiz 3
# Q1
library(AppliedPredictiveModeling)
data(segmentationOriginal)
names(segmentationOriginal)
seg <- segmentationOriginal
inTrain <- createDataPartition(y=seg$Case, p=0.7, list = FALSE)
training <- seg[seg$Case == "Train",]
testing <- seg[seg$Case == "Test",]
training <- seg[inTrain,]
testing <- seg[-inTrain,]
dim(training)
dim(testing)
set.seed(125)
modFit <- train(Class ~., method = "rpart", data = training)
print(modFit$finalModel)
# Plot
plot(modFit$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)

install.packages("https://cran.r-project.org/bin/windows/contrib/3.3/RGtk2_2.20.31.zip", repos=NULL)
library(rattle)
fancyRpartPlot(modFit$finalModel)

# Q3
library(pgmm)
data(olive)
olive = olive[,-1]
newdata = as.data.frame(t(colMeans(olive)))
newdata2 = data.frame(t(colMeans(olive))) # The same

modFit <- train(Area ~ ., data = olive, method = "rpart")
predict(modFit, newdata = newdata)
fancyRpartPlot(modFit$finalModel)

# Q4
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
train2 = sample(1:dim(SAheart)[1])
trainSA = SAheart[train,]
testSA = SAheart[-train,]

modFit <- train(chd ~ age+alcohol+obesity+tobacco+typea+ldl, method = "glm", family = "binomial",
                data = trainSA)

missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

missClass(testSA$chd, predict(modFit, trainSA))
missClass(testSA$chd, predict(modFit, testSA))

# Q5
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
set.seed(33833)
str(vowel.train)
str(vowel.test)
install.packages("randomForest")
library(randomForest)
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
modFit <- randomForest(y ~., data = vowel.train)
order(varImp(modFit),decreasing = TRUE)




############################# Week 4
### Regualized Regression
library(ElemStatLearn); data(prostate)
str(prostate)

small = prostate[1:5,]
lm(lpsa ~ .,data =small)


### Combining predictors
data(Wage)
Wage <- subset(Wage,select=-c(logwage))

# Create a building data set and validation set
inBuild <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
validation <- Wage[-inBuild,]; buildData <- Wage[inBuild,]

inTrain <- createDataPartition(y=buildData$wage,
                               p=0.7, list=FALSE)
training <- buildData[inTrain,]; testing <- buildData[-inTrain,]
# Build the model
mod1 <- train(wage ~.,method="glm",data=training)
mod2 <- train(wage ~.,method="rf",
              data=training, 
              trControl = trainControl(method="cv"),number=3)
# Predict on the testing set
pred1 <- predict(mod1,testing); pred2 <- predict(mod2,testing)
qplot(pred1,pred2,color=wage,data=testing)
# Fit a model that combines predictors
predDF <- data.frame(pred1,pred2,wage=testing$wage)
combModFit <- train(wage ~.,method="gam",data=predDF)    # gam method
combPred <- predict(combModFit,predDF)
# Testing errors
sqrt(sum((pred1-testing$wage)^2))
sqrt(sum((pred2-testing$wage)^2))
sqrt(sum((combPred-testing$wage)^2))
# Predict on validation data set
pred1V <- predict(mod1,validation); pred2V <- predict(mod2,validation)
predVDF <- data.frame(pred1=pred1V,pred2=pred2V)
combPredV <- predict(combModFit,predVDF)
# Evaluate on validation
sqrt(sum((pred1V-validation$wage)^2))
sqrt(sum((pred2V-validation$wage)^2))
sqrt(sum((combPredV-validation$wage)^2))


### FOrecasting
# Google data
library(quantmod)
from.dat <- as.Date("01/01/08", format="%m/%d/%y")
to.dat <- as.Date("12/31/13", format="%m/%d/%y")
getSymbols("GOOG", src="google", from = from.dat, to = to.dat)
head(GOOG)
# Summarize monthly and store as time series
mGoog <- to.monthly(GOOG)
googOpen <- Op(mGoog)
ts1 <- ts(googOpen,frequency=12)
plot(ts1,xlab="Years+1", ylab="GOOG")


### Unsupervised prediction
# Iris example ignoring species labels
data(iris)
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)
# Cluster with k-means
kMeans1 <- kmeans(subset(training,select=-c(Species)),centers=3)
training$clusters <- as.factor(kMeans1$cluster)
qplot(Petal.Width,Petal.Length,colour=clusters,data=training)
# Compare to real labels
table(kMeans1$cluster,training$Species)
# Build predictor
modFit <- train(clusters ~.,data=subset(training,select=-c(Species)),method="rpart")
table(predict(modFit,training),training$Species)
# Apply on test
testClusterPred <- predict(modFit,testing) 
table(testClusterPred ,testing$Species)


### Quiz 4
# Q1
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
head(vowel.train)
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)

modFit1 <- train(y ~., data = vowel.train, method = "rf")
modFit2 <- train(y ~., data = vowel.train, method = "gbm", verbose = FALSE)
predRf <- predict(modFit1, vowel.test)
predGbm <- predict(modFit2, vowel.test)

confusionMatrix(predRf, vowel.test$y)
confusionMatrix(predGbm, vowel.test$y)

predDf <- data.frame(predRf, predGbm, y=vowel.test$y)
confusionMatrix(predRf, predGbm)
sum(predDf$predRf[predDf$predRf == predDf$predGbm] ==
            predDf$y[predDf$predRf == predDf$predGbm])/
        sum(predDf$predRf == predDf$predGbm)

# Q2
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)
modFitRf <- train(diagnosis ~., data = training, method = "rf")
modFitGbm <- train(diagnosis ~., data = training, method = "gbm")
modFitLda <- train(diagnosis ~., data = training, method = "lda")

predRf <- predict(modFitRf, testing)
predGbm <- predict(modFitGbm, testing)
predLda <- predict(modFitLda, testing)

predDF <- data.frame(predRf, predGbm, predLda, diagnosis = testing$diagnosis)
modCombo <- train(diagnosis ~., data = predDF, method = "rf")
predCombo <- predict(modCombo, testing)

confusionMatrix(predRf, testing$diagnosis)
confusionMatrix(predGbm, testing$diagnosis)
confusionMatrix(predLda, testing$diagnosis)
confusionMatrix(predCombo, testing$diagnosis)

# Q3
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(233)
modLasso <- train(CompressiveStrength ~., data = training, method = "lasso")
library(elasticnet)
plot.enet(modLasso$finalModel, xvar = "penalty", use.color = TRUE)

# Q4
if (!file.exists("./Data")){
        dir.create("./Data")
}
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv",
              destfile = "./Data/gaData.csv")
https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv

library(lubridate)
dat = read.csv("./Data/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)
library(forecast)
modTs <- bats(tstrain)
fcast <- forecast(modTs, level = 95, h = dim(testing)[1])
sum(fcast$lower < testing$visitsTumblr & testing$visitsTumblr < fcast$upper) / 
        dim(testing)[1]

# Q5
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(325)
library(e1071)
mod_svm <- svm(CompressiveStrength ~ ., data = training)
pred_svm <- predict(mod_svm, testing)
accuracy(pred_svm, testing$CompressiveStrength)
































