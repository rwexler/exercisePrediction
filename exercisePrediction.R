exercisePrediction <- function() {
        
        # set working directory
        setwd("~/Google Drive/Data_Science/Machine_Learning/Course_Project/")
        
        # read training and testing sets
        library(RCurl)
        trainURL <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
        training <- read.csv(text = trainURL)
        testURL <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
        testing <- read.csv(text = testURL)
        
        # analyze size of training and testing sets
        dim(testing)[1]/dim(training)[1]*100 # 0.10% of data in testing set
        dim(testing)[1]+dim(training)[1]     # 19642 observations = medium sample size
        
        # create training and validation sets
        library(caret)
        set.seed(12345)
        inTrain <- createDataPartition(y = training$classe, p = 0.60, list = FALSE)
        training_new <- training[inTrain,]
        validation_new <- training[-inTrain,]
        dim(validation_new)[1]/dim(training_new)[1]*100 # 67% of data in testing set
        
        # explore training set
        dim(training_new)  # 11776 observations and 160 predictors
        head(training_new) # there are a lot of NA values
        # find columns with NA values
        colNA <- names(which(colSums(is.na(training_new)) < 1))
        training_new <- training_new[names(training_new) %in% colNA]
        validation_new <- validation_new[names(validation_new) %in% colNA] # also need to apply to testing
        # find columns with empty cells
        colEmpty <- names(which(colSums(training_new == "") < 1))
        training_new <- training_new[names(training_new) %in% colEmpty]
        validation_new <- validation_new[names(validation_new) %in% colEmpty] # also need to apply to testing
        
        # find near zero covariates
        nsv <- nearZeroVar(training_new, saveMetrics = TRUE) # none
        
        # LINEAR MODELS #
        
        # reformat training and validation data for using formulas
        training_new_formula <- training_new[,8:60]
        validation_new_formula <- validation_new[,8:60]
        
        # multiple linear regression model
        lmModel <- lm(classe ~ ., data = training_new_formula) # gyros_belt_x/z larges coefficients
        qplot(gyros_belt_x, gyros_belt_z, data = training_new_formula, colour = classe) # you can see some separation, not much
        
        ##### MODEL EVALUATION #####
        
        # TREES AND RANDOM FORESTS #
        
        # train decision tree
        library(rattle)
        # columns 1 through 7 identify the subject and date/time
        dcModel <- train(x = training_new[,8:59], y = training_new[,60], method = "rpart")
        fancyRpartPlot(dcModel$finalModel)
        # classe A is well-separated by pitch forearm < -34 and yaw belt >= 170
        # classe B is well-separated by pitch belt < -43
        # classe E is well-separated by roll belt >= 130
        
        # plot these relationships
        qplot(pitch_forearm, yaw_belt, data = training_new, colour = classe)
        qplot(yaw_belt, pitch_belt, data = training_new, colour = classe)
        qplot(roll_belt, classe, data = training_new)
        
        # evaluate performance of decision tree
        dcPrediction <- predict(dcModel, validation_new)
        confusionMatrix(dcPrediction, validation_new$classe) # accuracy = 0.4994, not so great
        
        # train random forest
        rfModel <- train(x = training_new[,8:59], y = training_new[,60], method = "rf", proximity = TRUE, ntree = 5, 
                         tuneLength = 1, trControl = trainControl(method = "none"))
        rfPrediction <- predict(rfModel, validation_new)
        confusionMatrix(rfPrediction, validation_new$classe) # accuracy = 0.9743, much better!
        
        # BOOSTING #
        
        # train gradient boosting model
        gbmModel <- train(x = training_new[,8:59], y = training_new[,60], method = "gbm",
                          tuneLength = 1, trControl = trainControl(method = "none"))
        gbmPrediction <- predict(gbmModel, validation_new[,8:59])
        confusionMatrix(gbmPrediction, validation_new$classe) # accuracy = 0.7539, worse
        
        # MODEL BASED PREDICTION #
        
        # train lda model
        ldaModel <- train(x = training_new[,8:59], y = training_new[,60], method = "lda",
                          tuneLength = 1, trControl = trainControl(method = "none"))
        ldaPrediction <- predict(ldaModel, validation_new[,8:59])
        confusionMatrix(ldaPrediction, validation_new$classe) # accuracy = 0.7039, worse
        
        # train naive bayes model
        library(e1071)
        nbModel <- naiveBayes(x = training_new[,8:59], y = training_new[,60])
        nbPrediction <- predict(nbModel, validation_new[,8:59])
        confusionMatrix(nbPrediction, validation_new$classe) # accuracy = 0.487, worse
        
        ##### PREPROCESSING ##### 
        
        # train random forest with standardized data
        #dcModelStd <- train(x = training_new[,8:59], y = training_new[,60], method = "rpart", 
        #                    preProcess = c("center", "scale"))
        #fancyRpartPlot(dcModelStd$finalModel)
        #dcStdPrediction <- predict(dcModelStd, validation_new)
        #confusionMatrix(dcStdPrediction, validation_new$classe) # accuracy = 0.4994, no improvement
        rfModelStd <- train(x = training_new[,8:59], y = training_new[,60], method = "rf", proximity = TRUE, ntree = 5, 
                         tuneLength = 1, trControl = trainControl(method = "none"), preProcess = c("center", "scale"))
        rfStdPrediction <- predict(rfModelStd, validation_new)
        confusionMatrix(rfStdPrediction, validation_new$classe) # accuracy = 0.9776, about the same
                
        # train random forest with principal components analysis
        #dcModelPCA <- train(x = training_new[,8:59], y = training_new[,60], method = "rpart",
        #                    preProcess = "pca")
        #fancyRpartPlot(dcModelPCA$finalModel)
        #dcPCAPrediction <- predict(dcModelPCA, validation_new)
        #confusionMatrix(dcPCAPrediction, validation_new$classe) # accuracy = 0.3824, worse
        # decreased performance may be due to changes in the default setting for resampling
        rfModelPCA <- train(x = training_new[,8:59], y = training_new[,60], method = "rf", proximity = TRUE, ntree = 5, 
                            preProcess = "pca", tuneLength = 1, 
                            trControl = trainControl(method = "none", preProcOptions = list(thresh = 0.95)))
        rfPCAPrediction <- predict(rfModelPCA, validation_new)
        confusionMatrix(rfPCAPrediction, validation_new$classe) # accuracy = 0.9058, decreased
        
        ##### CROSS VALIDATION #####
        
        # train random forest with cross validation
        #dcModelCV <- train(x = training_new[,8:59], y = training_new[,60], method = "rpart",
        #                   trControl = trainControl(method = "cv"))
        #fancyRpartPlot(dcModelCV$finalModel)
        #dcCVPrediction <- predict(dcModelCV, validation_new)
        #confusionMatrix(dcCVPrediction, validation_new$classe) # accuracy = 0.3676, worse
        # 5 folds
        rfModelCV5 <- train(x = training_new[,8:59], y = training_new[,60], method = "rf", proximity = TRUE, ntree = 5, 
                            tuneLength = 1, trControl = trainControl(method = "cv", number=5))
        rfCV5Prediction <- predict(rfModelCV5, validation_new)
        confusionMatrix(rfCV5Prediction, validation_new$classe) # accuracy = 0.9791, better
        
        # 10 folds
        rfModelCV10 <- train(x = training_new[,8:59], y = training_new[,60], method = "rf", proximity = TRUE, ntree = 5, 
                            tuneLength = 1, trControl = trainControl(method = "cv", number=10))
        rfCV10Prediction <- predict(rfModelCV10, validation_new)
        confusionMatrix(rfCV10Prediction, validation_new$classe) # accuracy = 0.9797, same
        
        ##### FINAL MODEL #####
        
        # random forest model with 10 trees and cross-validation with 5 folds
        rfFinalModel <- train(x = training_new[,8:59], y = training_new[,60], method = "rf", proximity = TRUE, ntree = 10, 
                              tuneLength = 1, trControl = trainControl(method = "cv", number=5))
        rfFinalPrediction <- predict(rfFinalModel, validation_new)
        confusionMatrix(rfFinalPrediction, validation_new$classe) # accuracy = 0.988, best!
        
        # apply to the test data
        testing_new <- testing[names(testing) %in% colNA]
        testing_new <- testing_new[names(testing_new) %in% colEmpty]
        predict(rfFinalModel, testing_new)
        
}