#Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(parallel)
library(doParallel)
library(tictoc)

#Data Import and Cleaning
gss<-read_sav("../data/GSS2016.sav")
gss_tbl<-gss %>% 
  drop_na(MOSTHRS) %>% 
  rename(`work hours`=MOSTHRS) %>% 
  select(-c(HRS1, HRS2)) %>% 
  select(where(~ mean(is.na(.)) < 0.75))
gss_tbl$`work hours`<-as.numeric(gss_tbl$`work hours`)

#Visualization
ggplot(gss_tbl, aes(x = `work hours`)) +
  geom_histogram()+
  labs(title = "Distribution of Work Hours",
       x = "Work Hours",
       y = "Frequency")

#Analysis
set.seed(0220)
holdout_indices <- createDataPartition(gss_tbl$`work hours`,
                                       p = .25,
                                       list = T)$Resample1
test_tbl <- gss_tbl[holdout_indices,]
training_tbl <- gss_tbl[-holdout_indices,]

training_folds <- createFolds(training_tbl$`work hours`)

#OLS regression model
ols_model <- train(`work hours` ~ ., 
                   data = training_tbl, 
                   method="lm",
                   na.action = na.pass,
                   preProcess = c("center","scale","zv","nzv","medianImpute"),
                   trControl = trainControl(method="cv", 
                                            number=10, 
                                            verboseIter=T, 
                                            indexOut = training_folds)
)

# Elastic Net model
elastic_net_model <- train(`work hours` ~ ., 
                           data = training_tbl, 
                           method="glmnet",
                           na.action = na.pass,
                           preProcess = c("center","scale","zv","nzv","medianImpute"),
                           trControl = trainControl(method="cv", 
                                                    number=10, 
                                                    verboseIter=T, 
                                                    indexOut = training_folds)
)

# Random Forest model
rf_model<-train(`work hours` ~ ., 
                training_tbl,  
                method="ranger",
                na.action = na.pass,
                tuneLength = 1,
                preProcess = c("center","scale","zv","nzv","medianImpute"),
                trControl = trainControl(method="cv", 
                                         number=10, 
                                         verboseIter=T, 
                                         indexOut = training_folds))


#eXtreme Gradient Boosting model
xgb_model <- train(`work hours` ~ ., 
                   training_tbl,  
                   method="xgbLinear",
                   na.action = na.pass,
                   tuneLength = 1,
                   preProcess = c("center","scale","zv","nzv","medianImpute"),
                   trControl = trainControl(method="cv", 
                                            number=10, 
                                            verboseIter=T, 
                                            indexOut = training_folds))



cv_ols <- max(ols_model$results$Rsquared)
cv_net<-max(elastic_net_model$results$Rsquared)
cv_rf<-max(rf_model$results$Rsquared)
cv_xgb<-max(xgb_model$results$Rsquared)


#CV prediction
ols_pred <- predict(ols_model, training_tbl, na.action=na.pass)
net_pred <- predict(elastic_net_model, training_tbl, na.action=na.pass)
rf_pred <- predict(rf_model, training_tbl, na.action=na.pass)
xgb_pred <- predict(xgb_model, training_tbl, na.action=na.pass)

# Evaluate holdout CV
ols_rmse<-cor(training_tbl$`work hours`, ols_pred)^2
net_rmse<-cor(training_tbl$`work hours`, net_pred)^2
rf_rmse<-cor(training_tbl$`work hours`, rf_pred)^2
xgb_rmse<-cor(training_tbl$`work hours`, xgb_pred)^2


#create a table 1
# Publication
make_it_pretty <- function (formatme) {
  formatme <- formatC(formatme, format="f", digits=2)
  formatme <- str_remove(formatme, "^0")
  return(formatme)
}

table1_tbl <- tibble(
  algo = c("regression","elastic net","random forests","xgboost"),
  cv_rqs = c(
    make_it_pretty(cv_ols),
    make_it_pretty(cv_net),
    make_it_pretty(cv_rf),
    make_it_pretty(cv_xgb)
  ),
  ho_rqs = c(
    make_it_pretty(ols_rmse),
    make_it_pretty(net_rmse),
    make_it_pretty(rf_rmse),
    make_it_pretty(xgb_rmse)
  )
)

table1_tbl

#Calculate the running time using function & tic and tock
time <- function(model, training_tbl, na.pass) {
  tic()  # Start the timer
  model_fit <- train(`work hours` ~ ., 
                     data = training_tbl, 
                     method = model,
                     trControl = trainControl(method="cv", 
                                              number=10, 
                                              verboseIter=T, 
                                              indexOut = training_folds),
                     preProcess = c("center","scale","zv","nzv","medianImpute"),
                     na.action = na.pass)
  time_taken <- toc(log = TRUE)  # End the timer and store the elapsed time
  return(time_taken$toc - time_taken$tic)  # Return the elapsed time
}

#Calculated the time for each model
model_lm_time <- time("lm", training_tbl, na.pass)
model_net_time <- time("glmnet", training_tbl, na.pass)
model_rf_time <- time("rf", training_tbl, na.pass)
model_xgb_time <- time("xgbLinear", training_tbl, na.pass)

##create a table2_tbl
# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
time_par <- function(model, training_tbl, na.pass) {
  tic()  # Start the timer
  model_fit <- train(`work hours` ~ ., 
                     data = training_tbl, 
                     method = model,
                     trControl = trainControl(method="cv", 
                                              number=10, 
                                              verboseIter=T, 
                                              indexOut = training_folds),
                     preProcess = c("center","scale","zv","nzv","medianImpute"),
                     na.action = na.pass)
  time_taken <- toc(log = TRUE)  # End the timer and store the elapsed time
  return(time_taken$toc - time_taken$tic)  # Return the elapsed time
}

#Calculated the time for each model
model_lm_time_par <- time_par("lm", training_tbl, na.pass)
model_net_time_par <- time_par("glmnet", training_tbl, na.pass)
model_rf_time_par <- time_par("rf", training_tbl, na.pass)
model_xgb_time_par <- time_par("xgbLinear", training_tbl, na.pass)

#Stopp arallization
stopCluster(cl)
registerDoSEQ()

#Create a table 2
table2_tbl <- tibble(
  algo = c("regression","elastic net","random forests","xgboost"),
  original = c(
    make_it_pretty(model_lm_time),
    make_it_pretty(model_net_time),
    make_it_pretty(model_rf_time),
    make_it_pretty(model_xgb_time)
  ),
  parallelized = c(
    make_it_pretty(model_lm_time_par),
    make_it_pretty(model_net_time_par),
    make_it_pretty(model_rf_time_par),
    make_it_pretty(model_xgb_time_par)
  )
)

table2_tbl

#Q1. Which models benefited most from parallelization and why?
## Based on the table random forest model presented the largest reduction in the running time when parallelized (220 --> 73.91 seconds).The random forest model's complex task works better under the paralleized condition.

#Q2. How big was the difference between the fastest and slowest parallelized model? Why?
###73.91 (Random forest model) - 2.23(Elastic net model). The gap between the running time is because of the complexity of the algorithm. 

#Q3. If your supervisor asked you to pick a model for use in a production model, which would you recommend and why? Consider both Table 1 and Table 2 when providing an answer.
##In terms of performance, the XGBoost model performs the best, followed by the random forests model. However, in terms of efficiency of model execution, the random forests model is the top performer, with XGBoost coming in next. Given that performance takes precedence over execution efficiency, I would recommend the random forests model to my supervisor.