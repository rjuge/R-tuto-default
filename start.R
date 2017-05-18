# Setting the data
rm(list = ls()) 
setwd("/Users/remi_juge/Downloads/code/") 
dung<-read.csv("credit.csv") 

#To delete unnecessary rows (but already did in the data table)
#dung[,c(1)] <- NULL
#dung=dung[,-1]


# Changing the labels of Y
dung$Y<- factor(dung$Y,levels = c("1", "0"),labels = c("Z2", "Z1")) 

#View 10 rows 
head(dung, n=10)

library(caret) 
set.seed(123) 

############## DATA CLEANING ################
dung <- dung[apply(dung[c(18:23)],1,function(z) any(z!=0)),]
dung <- dung[apply(dung[c(12:17)],1,function(z) any(z!=0)),]
#I obtain a 28497 obs. out of 30000 obs.

####### RANDOMLY SELECT 5000 ROWS #######
library(dplyr)
dung <- sample_n(dung, 500, replace = FALSE)


########### CHECKING THE CORRELATION MATRIX (IF NEEDED!!!) ############

correlationMatrix <- cor(dung[,c(1:23)])
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, names = TRUE, cutoff=0.5)
print(highlyCorrelated)
# On reduced dataset, it shows that there is no need to remove anything.
# From the original dataset, correlation matrix implies that its better to remove variables
# "X16" "X15" "X14" "X17" "X13" "X9"  "X10" "X8"  "X7" with the cutoff value of 0.5.
# Ideally, we could have chosen value 25, but the list of variables that should be removed increases to 15.
dung <- dung[,c(1:6,11,12,18:24)] # Take only needed variables

######### Divide the data into 2 equal sets: ##########

indxTrain <- createDataPartition(y = dung$Y,p = 0.5,list = FALSE) 
training <- dung[indxTrain,] 
testing <- dung[-indxTrain,] 


###########################################################################################################################
#ASSESSING ACCURACY ON TRAINING SET #######################################################################################

## Preparing training scheme
ctrl <- trainControl(method= "boot", number = 100, classProbs = TRUE, savePredictions = TRUE, summaryFunction = twoClassSummary) 

# TRAIN KNN
set.seed(234) 
knnFit <- train(Y~., data = training, method = "knn", trControl = ctrl, metric = "ROC", preProcess = c("center","scale"), tuneLength = 10)
knnFit
plot(knnFit, print.thres = 0.5, type="S")

# TRAIN LOGISTIC
set.seed(234) 
logistic <- train(Y~.,data=training,method="glm",family="binomial", trControl = ctrl, tuneLength = 10) 

# TRAIN NNET
set.seed(234) 
nnet=train(Y~.,data=training,method = "nnet",trControl=ctrl, preProcess = c("center","scale"), tuneLength=10,trace=FALSE)
png(filename="neural.png", width=700, height=700)
plot(nnet)
dev.off()
# TRAIN RANDOM FORESTS
set.seed(234) 
rf=train(Y~.,data=training,method = "rf",trControl=ctrl, preProcess = c("center","scale"), tuneLength=10)


# GETTING THE RESULTS
library(resample)
results <- resample(list(kNN=knnFit, Logistic = logistic, Nnet = nnet, Ranfor = rf))
summary(results)
bwplot(results)
dotplot(results)


##########################################################################################################################
#############ASSESSING ACCURACY ON TESTING SET ###########################################################################
library(party)
library(rpart)
################ TESTING KNN  ###################
knnPredict <- predict(knnFit,newdata = testing)
confusionMatrix(knnPredict, testing$Y)
################ TESTING LOG  ###################
logpred = predict(logistic, newdata=testing) 
confusionMatrix(logpred, testing$Y)
################ TESTING NNET  ##################
nnetpred = predict(nnet, newdata=testing)
confusionMatrix(nnetpred, testing$Y)
################ TESTING RFOR  ##################
## note: only 7 unique complexity parameters in default grid. Truncating the grid to 7 . 
pred = predict(rf, newdata=testing) 
confusionMatrix(pred, testing$Y)



################ ROC CURVES  ####################
png(filename="roc_curve_4_models.png", width=700, height=700)
## ROC KNN
library(ROCR)
library(ggplot2)
probknn <- predict(knnFit, testing, type='prob')
predknn <- prediction(probknn[, "Z2"],testing$Y)
perfknn <- performance(predknn, "tpr", "fpr")
plot(perfknn, col=2, main = "ROC for 4 models", xlab = "1-Specificity", ylab = "Sensitivity")   
abline(0,1)
legend(0.6, 0.6, c('kNN', 'Log', 'NNET','RFOR'), 2:5)
## ROC Logit
problog <- predict(logistic, testing, type="prob")
predlog <- prediction(problog[, "Z2"],testing$Y)
perflog <- performance(predlog, measure = "tpr", x.measure = "fpr")
plot(perflog, col=3, add=TRUE)   
abline(0,1)
## ROC NEURAL NETWRORKS
probnet <- predict(nnet, testing, type="prob")
prednet <- prediction(probnet[, "Z2"],testing$Y)
perfnet <- performance(prednet, measure = "tpr", x.measure = "fpr")
plot(perfnet, col=4, add=TRUE) 
abline(0,1)
## ROC RANDOM FORESTS
probrf <- predict(rf, testing, type="prob")
predrf <- prediction(probrf[, "Z2"],testing$Y)
perfrf <- performance(predrf, measure = "tpr", x.measure = "fpr")
plot(perfrf, col=5, add=TRUE) 
abline(0,1)
dev.off()


################ ACCURACY CURVES  ##################
png(filename="Accuracy-cutoff_4_models.png", width=700, height=700)
## ACCURACY KNN
acc.perf1 = performance(predknn, measure = "acc")
plot(acc.perf1, col=2)
legend(0.4, 0.6, c('kNN', 'Log', 'NNET','RFOR'), 2:5)
## ACCURACY LOG
acc.perf2 = performance(predlog, measure = "acc")
plot(acc.perf2, col=3, add=TRUE)
## ACCURACY NNET
acc.perf3 = performance(prednet, measure = "acc")
plot(acc.perf3, col=4, add=TRUE)
## ACCURACY RFOR
acc.perf4 = performance(predrf, measure = "acc")
plot(acc.perf4, col=5, add=TRUE)
dev.off()


############  GAIN CHARTS #####################
require("ROCR")
## GAIN CHART KNN
png(filename="Gain Chart - ALL together.png", width=700, height=700)
gain.knn = performance(predknn, "tpr", "rpp")
plot(gain.knn, col=2, lwd=2, main = "Gain charts for 4 models")
plot(x=c(0, 1), y=c(0, 1), type="l", col="black", lwd=2,
     ylab="True Positive Rate", 
     xlab="Rate of Positive Predictions")
gain.x = unlist(slot(gain.knn, 'x.values'))
gain.y = unlist(slot(gain.knn, 'y.values'))
lines(x=gain.x, y=gain.y, col=2, lwd=2)
legend(0.6, 0.6, c('kNN', 'Log', 'NNET','RFOR'), 2:5)

## GAIN CHART LOGIT
gain.log = performance(predlog, "tpr", "rpp")
plot(gain.log, col=3, lwd=2, add = TRUE)
#lines(x=gain.log.x, y=gain.log.y, col=3, lwd=2)

## GAIN CHART NEURAL NETWORKS
gain.net = performance(prednet, "tpr", "rpp")
plot(gain.net, col=4, lwd=2, add = TRUE)
#lines(x=gain.net.x, y=gain.net.y, col=4, lwd=2)

## GAIN CHART RANDOM FOREST
gain.rf = performance(predrf, "tpr", "rpp")
plot(gain.rf, col=5, lwd=2, add = TRUE)
#lines(x=gain.x.rf, y=gain.y.rf, col=5, lwd=2)
dev.off()


############# LIFT CHARTS 4 MODELS ################
## LIFT CHART KNN
png(filename="Lift Chart for 4 models.png", width=700, height=700)
lift.knn = performance(predknn, measure = "lift", x.measure = "rpp")
plot(lift.knn, col=2, main = "Lift charts for 4 models")   
abline(0,1)
legend(9, 7, c('kNN', 'Log', 'NNET','RFOR'), 2:5, plot = TRUE)
## LIFT CHART LOGIT
lift.log = performance(predlog, measure = "lift", x.measure = "rpp")
plot(lift.log, col = 3, add =TRUE)
## LIFT CHART NEURAL NETWORKS
lift.net = performance(prednet, measure = "lift", x.measure = "rpp")
plot(lift.net, col = 4, add =TRUE)
## LIFT CHART RANDOM FOREST
lift.rf = performance(predrf, measure = "lift", x.measure = "rpp")
plot(lift.rf, col = 5, add =TRUE)
dev.off()

############# FOR THE TABLE ##################################################
############# ACCURACY CUTOFF AUC #################
## ACC AUC KNN
auc.perf1 = performance(predknn, measure = "auc")
ind1 = which.max(slot(acc.perf1, "y.values")[[1]])
acc1 <- slot(acc.perf1, "y.values")[[1]][ind1]
acc1 <- round(acc1, digits = 3)
cutoff1 <- slot(acc.perf1,"x.values")[[1]][ind1]
cutoff1 <- round(cutoff1, digits = 3)
aucperf1 <- slot(auc.perf1, "y.values")[[1]]
aucperf1 <- round(aucperf1, digits = 3)
## ACC AUC LOG
auc.perf2 = performance(predlog, measure = "auc")
ind2 = which.max( slot(acc.perf2, "y.values")[[1]] )
acc2 <- slot(acc.perf2, "y.values")[[1]][ind2]
acc2 <- round(acc2, digits = 3)
cutoff2 <- slot(acc.perf2,"x.values")[[1]][ind2]
cutoff2 <- round(cutoff2, digits = 3)
aucperf2 <- slot(auc.perf2, "y.values")[[1]]
aucperf2 <- round(aucperf2, digits = 3)
## ACC AUC NEURAL NETWORKS
auc.perf3 = performance(prednet, measure = "auc")
ind3 = which.max( slot(acc.perf3, "y.values")[[1]] )
acc3 <- slot(acc.perf3, "y.values")[[1]][ind3]
acc3 <- round(acc3, digits = 3)
cutoff3 <- slot(acc.perf3,"x.values")[[1]][ind3]
cutoff3 <- round(cutoff3, digits = 3)
aucperf3 <- slot(auc.perf3, "y.values")[[1]]
aucperf3 <- round(aucperf3, digits = 3)
## ACC AUC RANDOM FOREST
auc.perf4 = performance(predrf, measure = "auc")
ind4 = which.max( slot(acc.perf4, "y.values")[[1]] )
acc4 <- slot(acc.perf4, "y.values")[[1]][ind4]
acc4 <- round(acc4, digits = 3)
cutoff4 <- slot(acc.perf4,"x.values")[[1]][ind4]
cutoff4 <- round(cutoff4, digits = 3)
aucperf4 <- slot(auc.perf4, "y.values")[[1]]
aucperf4 <- round(aucperf4, digits = 3)
########### ORGANIZING TABLE #########################
table <- matrix(c(acc1 ,acc2, acc3, acc4, cutoff1, cutoff2, cutoff3, cutoff4, aucperf1, aucperf2, aucperf3, round(aucperf4, digits = 3)),ncol=4,byrow=TRUE)
colnames(table) <- c("kNN","Logistic","Neural Networks", "Random Forest")
rownames(table) <- c("Accuracy","Cutoff","AUC")
table
library(gridExtra)
pdf(file = "TABLE.pdf")
grid.table(table)
dev.off()


