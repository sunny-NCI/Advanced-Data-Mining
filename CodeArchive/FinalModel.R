
##
## This is the code that is used to recreate our final model. This piece of code is also included in the CompleteWork.R script.
## For simplicity, we have extracted only the necessary code for this final model here.
##


#Clean the environment
rm(list=ls())

#Set the working directory
setwd('')

#Required packages
library(caret)
library(C50)
library(pROC)
library(ROCR)

#Read in the dataset
di <- read.csv("diabetic.csv")

###################################################### Data Preparation ####################################
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
di$time_in_hospital[(di$time_in_hospital >= 1 & di$time_in_hospital <= 3)] <- 'short'
di$time_in_hospital[(di$time_in_hospital >= 4 & di$time_in_hospital <= 6)] <- 'long'

di$time_in_hospital <- factor(di$time_in_hospital)

#################################### Feature Selection ##############################
cor_test <- subset(di, select = c('num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient'))
correlationMatrix <- cor(cor_test)
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
print(highlyCorrelated)

control <- trainControl(method="repeatedcv", number=10, repeats=3)

sample <- createDataPartition(di$time_in_hospital, p=0.10, list = F)
part <- di[sample, ]

model <- train(time_in_hospital~., data=part, method="lvq", preProcess="scale", trControl=control)
importance <- varImp(model, scale=FALSE)
print(importance)
plot(importance)

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(di[,1:8], di[,9], sizes=c(1:8), rfeControl=control)
print(results)
predictors(results)
plot(results, type=c("g", "o"))

######################## Removing the columns that are not important to our model #################
di$encounter_id <- NULL
di$patient_nbr <- NULL
di$readmitted <- NULL
di$admission_source_id <- NULL
di$admission_type_id <- NULL
di$discharge_disposition_id <- NULL
di$change <- NULL
di$metformin.rosiglitazone <- NULL
di$glimepiride.pioglitazone <- NULL
di$acetohexamide <- NULL
di$nateglinide <- NULL
di$miglitol <- NULL
di$chlorpropamide <- NULL 
di$glipizide.metformin <- NULL
di$glyburide.metformin <- NULL
di$repaglinide <- NULL
di$acarbose <- NULL
di$glimepiride <- NULL
di$rosiglitazone <- NULL 
di$gender <- NULL
di$tolazamide <- NULL
di$tolbutamide <- NULL
di$troglitazone <- NULL
di$metformin.pioglitazone <- NULL

##################################################
set.seed(123)

index <- createDataPartition(di$time_in_hospital, p=0.80, list=F)
training <- di[index, ]
testing <- di[-index, ]

###############################   C5.0 Model #################################

#We want a cross validation of 10 times
tuneParams <- trainControl(method = "cv", number = 10, savePredictions = 'final')

#Train the c5.0 model
starttime <- Sys.time()
paste0("It is ", starttime, ", starting C5.0 model now.")
c50Train <- train(training[,-3], training$time_in_hospital, method="C5.0", trControl=tuneParams, tuneLength=3)
endtime <- Sys.time()
paste0("It is now ", endtime, " Model took ", endtime-starttime)

#Predict on test set
c50Tpred <- predict(c50Train, testing)

#Evaluate the model
confusionMatrix(c50Tpred, testing$time_in_hospital)

pred <- prediction(as.numeric(c50Tpred), as.numeric(testing$time_in_hospital))
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)

