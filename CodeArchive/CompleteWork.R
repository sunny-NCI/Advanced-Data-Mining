
##
## This script is just to proof that the work we stated we did, we actually did.
## This is most of the code that we used to run all models that we produced.
## This code mostly uncommented and messy.
## If you're looking for the code for our final model, go to FinalModel.R
## 

#Clean the environment
rm(list=ls())

#Set the working directory
setwd('')

#Required packages
library(nnet)
library(e1071)
library(caret)
library(C50)
library(StatMatch)
library(pROC)
library(FactoMineR)
library(dplyr)
library(cluster)
library(factoextra)
library(ggfortify)
library(lattice)
library(plyr)
library(dplyr)
library(ROCR)

#Read in the dataset
di <- read.csv("diabetic.csv")

############################# Data Preparation #################################### 
di <- read.csv("diabetic.csv")
di <- di[ !(di$race == '?') ,]
di <- di[ !(di$time_in_hospital == 14),]
di <- di[ !(di$time_in_hospital == 13),]
di <- di[ !(di$time_in_hospital == 12),]
di <- di[ !(di$time_in_hospital == 11),]
di <- di[ !(di$time_in_hospital == 10),]
di <- di[ !(di$time_in_hospital == 9),]
di <- di[ !(di$time_in_hospital == 8),]
di <- di[ !(di$time_in_hospital == 7),]

di <- di[,-6]
di <- di[,-11]
di$diag_1 <- NULL
di$diag_2 <- NULL
di$diag_3 <- NULL
di$number_diagnoses <- NULL
di$payer_code <- NULL
di$examide <- NULL
di$citoglipton <- NULL
di$age <- factor(di$age, labels = c(1,2,3,4,5,6,7,8,9,10))
#di$time_in_hospital[(di$time_in_hospital > 0) & (di$time_in_hospital < 3)] <- 'short'
di$time_in_hospital[(di$time_in_hospital >= 1 & di$time_in_hospital <= 3)] <- 'short'
di$time_in_hospital[(di$time_in_hospital >= 4 & di$time_in_hospital <= 6)] <- 'long'

di$time_in_hospital <- factor(di$time_in_hospital)
di$encounter_id <- NULL
di$patient_nbr <- NULL
di$readmitted <- NULL
di$admission_source_id <- NULL
di$admission_type_id <- NULL
di$discharge_disposition_id <- NULL
di$change <- NULL

index <- createDataPartition(di$time_in_hospital, p=0.80, list=F)
training <- di[index, ]
testing <- di[-index, ]

training$metformin.rosiglitazone <- NULL
training$glimepiride.pioglitazone <- NULL
training$acetohexamide <- NULL
training$nateglinide <- NULL
training$miglitol <- NULL
training$chlorpropamide <- NULL 
training$glipizide.metformin <- NULL
training$glyburide.metformin <- NULL
training$repaglinide <- NULL
training$acarbose <- NULL
training$glimepiride <- NULL
training$rosiglitazone <- NULL 
training$gender <- NULL
training$tolazamide <- NULL
training$tolbutamide <- NULL
training$troglitazone <- NULL
training$metformin.pioglitazone <- NULL
testing$metformin.rosiglitazone <- NULL
testing$glimepiride.pioglitazone <- NULL
testing$acetohexamide <- NULL
testing$nateglinide <- NULL
testing$miglitol <- NULL
testing$chlorpropamide <- NULL 
testing$glipizide.metformin <- NULL
testing$glyburide.metformin <- NULL
testing$repaglinide <- NULL
testing$acarbose <- NULL
testing$glimepiride <- NULL
testing$rosiglitazone <- NULL 
testing$gender <- NULL
testing$tolazamide <- NULL
testing$tolbutamide <- NULL
testing$troglitazone <- NULL
testing$metformin.pioglitazone <- NULL

################ END of DATA PREPARATION ########################



############################ C5.0 ##########################################

model <- C5.0(time_in_hospital ~., data=training, trials=5)
tree <- predict(model, training, type = "class")
# argument newdata (training here) specifying the first place to look for explanatory variables to be used for prediction
confusionMatrix(tree, training$time_in_hospital)
treetest <- predict(model, testing, type = "class")
confusionMatrix(treetest, testing$time_in_hospital)

#########################Improving c5.0 model 
tuneParams <- trainControl(method = "cv", number = 10, savePredictions = 'final')
c50Train <- train(training[,-3], training$time_in_hospital, method="C5.0", trControl=tuneParams, tuneLength=3)


c50Tpred <- predict(c50Train, testing)
confusionMatrix(c50Tpred, testing$time_in_hospital)
pred <- prediction(as.numeric(c50Tpred), as.numeric(testing$time_in_hospital))
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)

############ Logit ###################################################################


#logit <- multinom(time_in_hospital ~., data = training)
logitpredict <- predict(logit, testing, type = "class")
confusionMatrix(logitpredict, testing$time_in_hospital)
importance <- varImp(logit, scale=FALSE)

auc <- performance(logitpredict, "auc")

##########################   SVM  ###################################################

start_time <- Sys.time()
model <- svm(time_in_hospital ~., data=training, kernel="linear")
svm <- predict(model, training)
confusionMatrix(svm, training$time_in_hospital)
end_time <- Sys.time()
RuntimeM <- end_time - start_time


###################### Unsupervised Learning - FAMD Clustering ##################################
set.seed(1337)
training$time_in_hospital <- NULL
training$age <- as.numeric(levels(training$age))[training$age]
index2 <- sample(1:nrow(training), nrow(training) * 0.60, replace=FALSE)
sample <- training[index2,]


library(graphics)
famd <- FAMD(training, ncp = 5, graph = TRUE, sup.var = NULL, ind.sup = NULL, axes = c(1,2), row.w = NULL, tab.comp = NULL)
plot(famd, choix = c("ind","var","quanti","quali"), axes = c(1, 2), lab.var = TRUE, lab.ind = TRUE, habillage = "none", col.lab = FALSE, col.hab = NULL, invisible = NULL, lim.cos2.var = 0., xlim = NULL, ylim = NULL, title = NULL, palette=NULL, autoLab = c("auto","yes","no"), new.plot = FALSE, select = NULL, unselect = 0.7, shadowtext = FALSE)



############################### Random Forest ########################################
library(randomForest)
set.seed(1337)
training$time_in_hospital <- NULL
training$age <- as.numeric(levels(training$age))[training$age]
index2 <- sample(1:nrow(training), nrow(training) * 0.40, replace=FALSE)
sample <- training[index2,]

starttime <- Sys.time()
paste0("It is ", starttime, ", starting random forest model now.") 
forest <-randomForest(time_in_hospital~., data=training, importance=TRUE, ntree=3000)
#rf <-predict(forest, testing, type = "class")
endtime <- Sys.time()
paste0("It is now ", endtime, " Model took ", endtime-starttime)
rfpred <- predict(forest, testing)
confusionMatrix(rfpred, testing$time_in_hospital)

library(rpart)
starttime <- Sys.time()
paste0("It is ", starttime, ", starting rpart model now.") 
Rpart <-rpart(time_in_hospital~., data=training,method="class", control=rpart.control(minsplit=3, cp=0))
endtime <- Sys.time()
paste0("It is now ", endtime, " Model took ", endtime-starttime)


plot(Rpart)
text(Rpart)
predict(Rpart, testing$time_in_hospital,type = "class")
###################### Unsupervised Learning - FUZZY Clustering ##################################
set.seed(1337)
training$time_in_hospital <- NULL
training$age <- as.numeric(levels(training$age))[training$age]
index2 <- createDataPartition(di$time_in_hospital, p=0.40, list=F)
sample <- training[index2,]

starttime <- Sys.time()
paste0("It is ", starttime, ", starting clustering now.") 
fanny(sample, 2, metric = "euclidean", stand = FALSE)
fviz_cluster(fanny, ellipse.type = "norm", repel = TRUE, palette = "jco", ggtheme = theme_minimal(), legend = "right")
endtime <- Sys.time()
paste0("It is now ", endtime, " Model took ", endtime-starttime)


